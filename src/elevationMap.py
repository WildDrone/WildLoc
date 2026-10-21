from pyproj.transformer import Transformer  # noqa: I001
from pyproj import CRS
import rasterio
import numpy as np
import pyvista as pv
import rioxarray
import os
from pathlib import Path

scriptPath = os.path.dirname(os.path.realpath(__file__))


class Terrain:
    def __init__(self):
        # Load DEM
        self.loadSurface()

    def loadSurface(self):
        # Load DEM
        data = rioxarray.open_rasterio(self.mapPath)
        values = np.asarray(data)

        values[values == -9999.0] = np.nan

        # Make a mesh
        xx, yy = np.meshgrid(data["x"], data["y"])
        zz = values.reshape(xx.shape)

        # Compute crop indices
        self.eastingNW, self.northingNW = self.convertToMapCRS(
            self.latNW, self.lonNW, self.COORDINATE_SYSTEM
        )
        self.eastingSE, self.northingSE = self.convertToMapCRS(
            self.latSE, self.lonSE, self.COORDINATE_SYSTEM
        )
        xNW = self.find_nearest(data["x"], self.eastingNW)
        xSE = self.find_nearest(data["x"], self.eastingSE)
        yNW = self.find_nearest(data["y"], self.northingNW)
        ySE = self.find_nearest(data["y"], self.northingSE)

        # Make Structured Grid from cropped mesh
        self.mesh = pv.StructuredGrid(
            xx[yNW:ySE, xNW:xSE], yy[yNW:ySE, xNW:xSE], zz[yNW:ySE, xNW:xSE]
        )
        self.mesh["data"] = zz[yNW:ySE, xNW:xSE].ravel(order="F")

        # Extract surface
        self.surface = self.mesh.extract_surface(algorithm="dataset_surface")

    def findIntersectionWithSurface(self, cameraOrigin, rayDirection, rayLength=2000):
        # cameraOrigin: position of the camera focal point expressed in WGS84
        # rayDirection: vector from camera origin to vPOI expressed in NED

        # Convert ray from NED to ENU and normalize
        rayDirection = np.array([rayDirection[1], rayDirection[0], -rayDirection[2]])
        rayDirection = rayDirection / np.linalg.norm(rayDirection)
        rayDirection = rayDirection.ravel()

        # Convert cameraOrigin from WGS84 to UK frame
        camEast, camNorth = self.convertToMapCRS(
            cameraOrigin[0], cameraOrigin[1], self.COORDINATE_SYSTEM
        )
        cameraOrigin = np.array([camEast, camNorth, cameraOrigin[2]])

        # Compute intersection
        intersection, _ = self.surface.ray_trace(
            cameraOrigin,
            cameraOrigin + rayLength * rayDirection,
            first_point=True,
            plot=False,
        )
        if len(intersection) == 0:
            print("WARNING: No intersection found, returning default values")
            # raise ValueError("No intersection found with the surface. Check the camera position and ray direction.")
            return 0, 0, 0

        # Convert to WGS84
        lat, lon = self.convertFromMapCRS(
            intersection[0], intersection[1], self.COORDINATE_SYSTEM
        )
        h = intersection[2]
        # print(f"Altitude {h}")

        return lat, lon, h

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def to_easting_northing(self, lat, lon):
        source_proj = CRS.from_epsg(4326)
        target_proj = CRS.from_epsg(27700)
        transformation = Transformer.from_crs(source_proj, target_proj, always_xy=True)
        easting, northing = transformation.transform(lon, lat)
        return easting, northing

    def convertToMapCRS(self, lat, lon, mapCRS):
        WGS84 = CRS.from_epsg(4326)
        transformation = Transformer.from_crs(WGS84, mapCRS, always_xy=True)
        easting, northing = transformation.transform(lon, lat)
        return easting, northing

    def convertFromMapCRS(self, easting, northing, mapCRS):
        WGS84 = CRS.from_epsg(4326)
        transformation = Transformer.from_crs(mapCRS, WGS84, always_xy=True)
        lon, lat = transformation.transform(easting, northing)
        return lat, lon

    def to_lat_lon(self, easting, northing):
        source_proj = CRS.from_epsg(27700)
        target_proj = CRS.from_epsg(4326)
        transformation = Transformer.from_crs(source_proj, target_proj, always_xy=True)
        lon, lat = transformation.transform(easting, northing)
        return lat, lon

    def import_terrain(self, terrain_path):
        dem_data = rasterio.open(terrain_path)
        dem_array = dem_data.read(1)
        transform = dem_data.transform
        dem_height, dem_width = dem_array.shape
        dem_array = dem_array.astype(np.float32)
        dem_data.close()
        return dem_array, transform, dem_height, dem_width

    def transform_to_grid(self, x_coordinate, y_coordinate, transform):
        inv_transform = ~transform
        grid_x, grid_y = [
            int(round(coord)) for coord in inv_transform * (x_coordinate, y_coordinate)
        ]
        return grid_x, grid_y

    def getElevation(self, lat, lon, method="nearest"):
        if method == "nearest":
            # Fastest but less accurate
            easting, northing = self.convertToMapCRS(lat, lon, self.COORDINATE_SYSTEM)
            x = self.find_nearest(self.mesh.points[:, 0], easting)
            y = self.find_nearest(self.mesh.points[:, 1], northing)
            index = y * self.mesh.dimensions[0] + x
            elevation = self.mesh["data"][index]
        elif method == "rayTrace":
            # More accurate but slower
            cameraOrigin = np.array([lat, lon, self.mesh.bounds[4]])
            rayDirection = np.array([0, 0, -1])
            lat, lon, elevation = self.findIntersectionWithSurface(
                cameraOrigin,
                rayDirection,
                rayLength=self.mesh.bounds[5] - self.mesh.bounds[4] + 1,
            )
        elif method == "interpolate":
            # TODO: implement interpolation
            # How it should be done, but can't find a way to do it...
            raise NotImplementedError("Interpolation method not implemented yet")
            easting, northing = self.convertToMapCRS(lat, lon, self.COORDINATE_SYSTEM)
            point = np.array([[easting, northing, 1], [easting + 1, northing, 2]])
            point_cloud = pv.PolyData(point)
            sampledMesh = point_cloud.interpolate(self.surface)
            elevation = sampledMesh["data"]
        else:
            raise ValueError(f"Method {method} not supported")

        return elevation


class Fenswood(Terrain):
    def __init__(self):
        # Elevation data source
        self.mapPath = Path("data/maps/dem/fenswood.tif")
        self.COORDINATE_SYSTEM = CRS.from_epsg(27700)

        # Cropping (none here)
        self.latNW = 70
        self.lonNW = -90
        self.latSE = -70
        self.lonSE = 90

        super().__init__()


class OlPejetaZebraHerd2(Terrain):
    def __init__(self):
        # Elevation data source
        self.mapPath = Path("data/maps/dem/opcWaterHole_ODMdem_lowRes.tif")
        self.COORDINATE_SYSTEM = CRS.from_epsg(32637)

        # Cropping (none here)
        self.latNW = 70
        self.lonNW = -90
        self.latSE = -70
        self.lonSE = 90

        super().__init__()


if __name__ == "__main__":
    map = OlPejetaZebraHerd2()
    map.mesh.plot()
