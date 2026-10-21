import numpy as np # noqa: I001
from scipy.spatial.transform import Rotation as R
import pymap3d
from tqdm import tqdm

from elevationMap import Terrain
from dataStructure import LocalisedObject, LabelledDroneImage, LocalisedDroneImage, geoLoc
from camera import CameraAll

class ObjectPosition:
    def __init__(self, USE_ELEVATION : bool = False, MAP : Terrain = None):
        self.USE_ELEVATION = USE_ELEVATION
        self.MAP = MAP
        if USE_ELEVATION:
            if MAP is None:
                raise ValueError("MAP must be provided when USE_ELEVATION is True")
            self.getObjectPosition = self.getObjectPosition_elevPixel
        else:
            self.getObjectPosition = self.getObjectPosition_flatPixel
        self.altHome = 0
        self.camera = CameraAll()

    def getLocalisedDroneImages(self, labelledImages : list[LabelledDroneImage], altFudge : float = 0, pitchFudge : float = 0, yawFudge : float = 0):
        self.setAltHome(labelledImages[0].param.homeLoc, altFudge)
        localisedDroneImages = []
        for labelledImage in tqdm(labelledImages, desc="Localising objects"):
            localisedObjects = {}
            for id, obj in labelledImage.objects.items():
                # Undistort the image coordinates
                x, y = obj.imageX, obj.imageY
                x, y = self.camera.undistortCoords(labelledImage, x, y)
                h = labelledImage.image.shape[0]
                w = labelledImage.image.shape[1]
                dz = labelledImage.param.digitalZoomRatio
                x = (w - w/dz)/2 + x/dz
                y = (h - h/dz)/2 + y/dz
                position = self.getObjectPosition(x,
                                                  y,
                                                  self.camera.getFocalLength(labelledImage),
                                                  self.camera.getCx(labelledImage),
                                                  self.camera.getCy(labelledImage),
                                                  labelledImage.param.GimbalPitchDegree + pitchFudge,
                                                  labelledImage.param.GimbalYawDegree + yawFudge,
                                                  labelledImage.param.GPSLatitude,
                                                  labelledImage.param.GPSLongitude,
                                                  labelledImage.param.GPSAltitudeRelative
                                                  )
                if self.USE_ELEVATION:
                    loc = geoLoc(position["animalLat"], position["animalLon"], position["animalAlt"])
                else:
                    loc = geoLoc(position["animalLat"], position["animalLon"], 0)

                localisedObject = LocalisedObject(obj, loc)
                localisedObjects[id] = localisedObject
            localisedDroneImage = LocalisedDroneImage(labelledImage, localisedObjects)
            localisedDroneImages.append(localisedDroneImage)
        return localisedDroneImages

    def getObjectPosition_flatPixel(self, x_image, y_image, f, cx, cy, gb_pitch, gb_yaw, lat_drone, lon_drone, alt_drone):
        # Adjust altitude with home altitude
        alt_drone = alt_drone + self.altHome  

        # Position of the object in the image expressed in the camera frame
        r_c = np.array([[f],[x_image - cx],[y_image - cy]])

        # Rotation matrix from camera frame to world frame
        C_cw = R.from_euler('ZY', [gb_yaw, gb_pitch], degrees=True).as_matrix()

        # Computing the intersection with ground
        alpha = alt_drone/np.matmul(C_cw,r_c)[2][0]
        r_w = alpha*np.matmul(C_cw,r_c) + np.array([[0], [0], [-alt_drone]])
        
        # Compute position in earth frame
        animalPosition = pymap3d.ned2geodetic(r_w[0][0], r_w[1][0], 0, lat_drone, lon_drone, 0)[0:2]

        results = {"animalLat":animalPosition[0], "animalLon":animalPosition[1], "northingFromDrone":r_w[0][0], "eastingFromDrone":r_w[1][0]}
        return results

    def getObjectPosition_elevPixel(self, x_image, y_image, f, cx, cy, gb_pitch, gb_yaw, lat_drone, lon_drone, alt_drone_Home):
        # print(f"Call to getObjectPosition_elevPixel with {x_image}, {y_image}, {f}, {cx}, {cy}, {gb_pitch}, {gb_yaw}, {lat_drone}, {lon_drone}, {alt_drone_Home}")
        # Position of the object in the image expressed in the camera frame
        r_c = np.array([[f],[x_image - cx],[y_image - cy]])

        # Rotation matrix from camera frame to world frame
        C_cw = R.from_euler('ZY', [gb_yaw, gb_pitch], degrees=True).as_matrix()

        # Find intersection with sruface
        alt_drone_DEM = alt_drone_Home + self.altHome
        animalLat, animalLon, animalAlt = self.MAP.findIntersectionWithSurface(np.array([lat_drone, lon_drone, alt_drone_DEM]), np.matmul(C_cw,r_c))

        results = {"animalLat":animalLat, "animalLon":animalLon, "animalAlt":animalAlt}
        return results
    
    def setAltHome(self, homeLoc : geoLoc, altFudge : float = 0):
        if self.USE_ELEVATION is False:
            self.altHome = 0
        else:
            print(f"Home location: {homeLoc}")
            self.altHome = self.MAP.getElevation(homeLoc.lat, homeLoc.lon, method="rayTrace")
        
        if altFudge != 0:
            self.altHome += altFudge
            print("\033[33mWARNING: added fudge factor on home altitude!\033[0m")

        print(f"Home altitude set to {self.altHome}")

    