import pymap3d as pm # noqa: I001
from cv2.typing import MatLike
import cv2
from pyproj.transformer import Transformer
from pyproj import CRS
import numpy as np

class geoLoc:
    def __init__(self, lat, lon, alt=None):
        self.lat = float(lat)
        self.lon = float(lon)
        if alt is not None:
            alt = float(alt)
        self.alt = alt
        self._validate()

    def _validate(self):
        if not (-90 <= self.lat <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if not (-180 <= self.lon <= 180):
            raise ValueError("Longitude must be between -180 and 180 degrees")
        
    def offset(self, east, north, up=None):
        if self.alt:
            lat, lon, alt = pm.enu2geodetic(east, north, up or 0, self.lat, self.lon, self.alt)
            return geoLoc(lat, lon, alt)
        else:
            lat, lon, _ = pm.enu2geodetic(east, north, 0, self.lat, self.lon, 0)
            return geoLoc(lat, lon)
        
    def distTo(self, other : "geoLoc", hzOnly=False):
        if hzOnly:
            _, _, srange = pm.geodetic2aer(self.lat, self.lon, 0, other.lat, other.lon, 0)
        else:
            _, _, srange = pm.geodetic2aer(self.lat, self.lon, self.alt or 0, other.lat, other.lon, other.alt or 0)
        return srange
    
    def aer(self, other : "geoLoc", hzOnly=False):
        if hzOnly:
            az, el, r = pm.geodetic2aer(self.lat, self.lon, 0, other.lat, other.lon, 0)
        else:
            az, el, r = pm.geodetic2aer(self.lat, self.lon, self.alt or 0, other.lat, other.lon, other.alt or 0)
        return az, el, r
    
    def __str__(self):
        if self.alt is None:
            return f"Latitude: {self.lat:.6f}, Longitude: {self.lon:.6f}, Google Maps: https://www.google.com/maps/search/{self.lat:.6f},{self.lon:.6f}"
        else:
            return f"Latitude: {self.lat:.6f}, Longitude: {self.lon:.6f}, Altitude: {self.alt:.1f}, Google Maps: https://www.google.com/maps/search/{self.lat:.6f},{self.lon:.6f}"
    
    def convertToCRS(self, destCRS):
        WGS84 = CRS.from_epsg(4326)
        transformation = Transformer.from_crs(WGS84, destCRS, always_xy=True)
        easting, northing = transformation.transform(self.lon, self.lat)
        return easting, northing

class Object:
    def __init__(self, id : int, imageX : int, imageY : int, description: str = "", color: tuple[int, int, int] = (255, 0, 0)):   
        self.id = id
        self.imageX = imageX
        self.imageY = imageY
        self.description = description
        self.color = color

class LocalisedObject(Object):    
    def __init__(self, obj : Object, loc : geoLoc):
        super().__init__(obj.id, obj.imageX, obj.imageY, obj.description)
        self.loc = loc

class ImageParam:
    def __init__(self, metadata: dict = {}):
        self.metadata = metadata
        if "XMP:DroneModel" in self.metadata:
            self.droneType = self.metadata.get("XMP:DroneModel", None)
        elif "OSD.droneType" in self.metadata:
            self.droneType = self.metadata.get("OSD.droneType", None)

        if self.droneType in ["M3E", "M3T"]:
            self.loadM3E()
        elif self.droneType in ["Mini 3"]:
            self.loadMini3()
        else:
            raise ValueError(f"Unknown drone type {self.droneType}")
        
    def loadM3E(self):
        self.timestamp = self.metadata.get("imageTimestamp", None)
        self.fileName = self.metadata.get("File:FileName", None)
        self.imageWidth = self.metadata.get("File:ImageWidth", None)
        self.imageHeight = self.metadata.get("File:ImageHeight", None)
        self.digitalZoomRatio = self.metadata.get("EXIF:DigitalZoomRatio", None)
        self.focalLengthIn35mmFormat = self.metadata.get("EXIF:FocalLengthIn35mmFormat", None)
        self.GPSStatus = self.metadata.get("XMP:GpsStatus", None)
        self.GPSLatitude = self.metadata.get("XMP:GPSLatitude", None)
        self.GPSLongitude = self.metadata.get("XMP:GPSLongitude", None)
        self.GPSAltitudeAbsolute = self.metadata.get("XMP:AbsoluteAltitude", None)
        self.GPSAltitudeRelative = self.metadata.get("XMP:RelativeAltitude", None)
        self.GPSLoc = geoLoc(self.GPSLatitude, self.GPSLongitude, self.GPSAltitudeRelative)
        self.GimbalRollDegree = self.metadata.get("XMP:GimbalRollDegree", None)
        self.GimbalPitchDegree = self.metadata.get("XMP:GimbalPitchDegree", None)
        self.GimbalYawDegree = self.metadata.get("XMP:GimbalYawDegree", None)
        self.focalLengthPixel = 730.14555062*self.digitalZoomRatio*7
        self.homeLat = self.metadata.get("homeLat", None)
        self.homeLon = self.metadata.get("homeLon", None)
        self.homeLoc = geoLoc(self.homeLat, self.homeLon)

    def loadMini3(self):
        self.timestamp = self.metadata.get("imageTimestamp", None)
        self.fileName = self.metadata.get("File:FileName", None)
        self.imageWidth = self.metadata.get("QuickTime:SourceImageWidth", None)
        self.imageHeight = self.metadata.get("QuickTime:SourceImageHeight", None)
        self.digitalZoomRatio = 1
        self.focalLengthIn35mmFormat = None
        self.GPSStatus = None
        self.GPSnum = self.metadata.get("OSD.gpsNum", None)
        self.GPSLatitude = self.metadata.get("OSD.latitude", None)
        self.GPSLongitude = self.metadata.get("OSD.longitude", None)
        self.GPSAltitudeAbsolute = self.metadata.get("OSD.altitude [ft]", None) * 0.3048
        self.GPSAltitudeRelative = self.metadata.get("OSD.height [ft]", None) * 0.3048
        self.GPSLoc = geoLoc(self.GPSLatitude, self.GPSLongitude, self.GPSAltitudeRelative)
        self.GimbalRollDegree = self.metadata.get("GIMBAL.roll", None)
        self.GimbalPitchDegree = self.metadata.get("GIMBAL.pitch", None)
        self.GimbalYawDegree = self.metadata.get("GIMBAL.yaw", None)
        self.focalLengthPixel = 872.1972189073363
        self.homeLat = self.metadata.get("HOME.latitude", None)
        self.homeLon = self.metadata.get("HOME.longitude", None)
        self.homeLoc = geoLoc(self.homeLat, self.homeLon)
    
    def __str__(self):
        attributes = [a for a in vars(self) if a != "metadata"]
        str = ""
        for a in attributes:
            str += f"{a}: {getattr(self, a)}, type: {type(getattr(self, a))}\n"
        return str
    
class DroneImage:
    def __init__(self, image : MatLike, param : ImageParam):
        self.image = image
        self.param = param

    def __str__(self):
        return self.param.__str__()
    
    def show(self, winName : str, *args):
        cv2.imshow(winName, self.image)
        
    def noImage(self):
        self.image = None

    def unzoom(self, inPlace : bool = False):
        # Create a black image with the same size as image
        unzoomedImage = np.zeros((self.param.imageHeight, self.param.imageWidth, 3), dtype=np.uint8)

        # Subsample the orginal image
        subsampledImage = cv2.resize(self.image, (0,0), fx=1/self.param.digitalZoomRatio, fy=1/self.param.digitalZoomRatio)

        # Copy the subsampled image at the center of the black image
        self.unzoomXOffset = (unzoomedImage.shape[1] - subsampledImage.shape[1]) // 2
        self.unzoomYOffset = (unzoomedImage.shape[0] - subsampledImage.shape[0]) // 2
        unzoomedImage[self.unzoomYOffset:self.unzoomYOffset + subsampledImage.shape[0], self.unzoomXOffset:self.unzoomXOffset + subsampledImage.shape[1]] = subsampledImage
        
        if inPlace:
            self.image = unzoomedImage
        else:
            return unzoomedImage
    
class LabelledDroneImage(DroneImage):
    def __init__(self, image : DroneImage, objects : dict[int, Object]):
        super().__init__(image.image, image.param)
        self.objects = objects

    def show(self, winName : str, showID : bool = True, showDescription : bool = True):
        circleSize = 3
        tmpImg = self.image.copy()
        for obj in self.objects.values():
            cv2.circle(tmpImg, (obj.imageX, obj.imageY), circleSize, obj.color, 2)
            if showID:
                cv2.putText(tmpImg, f'ID: {obj.id}', (obj.imageX + 5, obj.imageY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj.color, 2)
            if showDescription:
                cv2.putText(tmpImg, f'Desc: {obj.description}', (obj.imageX + 25, obj.imageY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj.color, 2)
        
        cv2.imshow(winName, tmpImg)

    def flatten(self):
        # Creats a list of LabbeledDroneImage with only one object
        return [LabelledDroneImage(self, {obj.id: obj}) for obj in self.objects.values()]
    
    def filter(self, ids : list[int]):
        return LabelledDroneImage(self, {id: obj for id, obj in self.objects.items() if id in ids})

    def areIdUnique(self):
        return len(self.objects) == len({obj.id for obj in self.objects.values()})
    
    def sortById(self):
        return LabelledDroneImage(self, {id: obj for id, obj in sorted(self.objects.items(), key=lambda item: item[0])})
    
    def unzoom(self, inPlace : bool = False):
        imgUnzoomed = super().unzoom(inPlace = False)
        newObjs = {}
        for obj in self.objects.values():
            newObj = obj
            newObj.imageX = int(obj.imageX / self.param.digitalZoomRatio) + self.unzoomXOffset
            newObj.imageY = int(obj.imageY / self.param.digitalZoomRatio) + self.unzoomYOffset
            newObjs[obj.id] = newObj

        if inPlace:
            self.image = imgUnzoomed
            self.objects = newObjs
            self.param.digitalZoomRatio = 1
        else:
            newParam = self.param
            newParam.digitalZoomRatio = 1
            return LabelledDroneImage(DroneImage(imgUnzoomed, newParam), newObjs)

    
    @staticmethod
    def getMatchedFlatImages(mainImages : list["LabelledDroneImage"], refImages : list["LabelledDroneImage"], selected : list[int] = []):
        # If selected is empty, select the intersection of IDs

        # Filter the images by ID
        if selected == []:
            mainImages = [imgMain.filter(list(set(imgMain.objects.keys()).intersection(set(imgRef.objects.keys())))) for imgMain, imgRef in zip(mainImages, refImages)]
            refImages = [imgRef.filter(list(set(imgMain.objects.keys()).intersection(set(imgRef.objects.keys())))) for imgMain, imgRef in zip(mainImages, refImages)]
        else:
            mainImages = [img.filter(selected) for img in mainImages]
            refImages = [img.filter(selected) for img in refImages]        

        # Check if the IDs are unique
        if not all(img.areIdUnique() for img in mainImages):
            raise ValueError("IDs in mainImages are not unique")
        if not all(img.areIdUnique() for img in refImages):
            raise ValueError("IDs in refImages are not unique")
        
        # Sort the images by ID
        mainImages = [img.sortById() for img in mainImages]
        refImages = [img.sortById() for img in refImages]

        # Flatten the images
        mainImages = [img.flatten() for img in mainImages]
        refImages = [img.flatten() for img in refImages]

        # Check if the number of images are the same
        if len(mainImages) != len(refImages):
            raise ValueError("Length of mainImages and refImages must be the same")
        
        # Combine the images
        mainImages = [img for sublist in mainImages for img in sublist]
        refImages = [img for sublist in refImages for img in sublist]
        return mainImages, refImages

class LocalisedDroneImage(LabelledDroneImage):
    def __init__(self, image : DroneImage, objects : dict[int, LocalisedObject]):
        super().__init__(image, objects)
        
class ColorPalette:
    def __init__(self):
        self.colorPalette = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (0, 255, 255),    # Cyan
            (255, 0, 255),    # Magenta
            (128, 0, 128),    # Purple
            (128, 0, 0),      # Maroon
            (128, 128, 0),    # Olive
            (0, 128, 0),      # Dark Green
            (0, 0, 128),      # Navy
            (255, 165, 0),    # Orange
            (255, 192, 203),  # Pink
            (210, 105, 30),   # Chocolate
            (75, 0, 130),     # Indigo
            (173, 255, 47),   # Green Yellow
            (192, 192, 192),  # Silver
            (255, 20, 147)    # Deep Pink
        ]
    
    def getColor(self, ID, maxIntensity=255, returnPairs=False):
        if returnPairs:
            ID = ID // 2
        if maxIntensity != 255:
            return tuple([c*maxIntensity/255 for c in self.colorPalette[ID % len(self.colorPalette)]])
        else:
            return self.colorPalette[ID % len(self.colorPalette)]
    
