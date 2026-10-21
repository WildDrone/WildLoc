import cv2 # noqa: I001
from exiftool import ExifToolHelper
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._libs.tslibs.timedeltas import Timedelta
from dataStructure import DroneImage, ImageParam
import os
import numpy as np

class DjiImageReader:
    def __init__(self):
        pass

    def getDroneImage(self, imagePath : str) -> DroneImage:
        # Load image 
        image = cv2.imread(imagePath)
        
        # Extract metadata from image
        with ExifToolHelper() as et:
            metadata = et.get_metadata(imagePath)[0]

        # Convert metadata values to numeric
        def convertValuesToNumeric(inputDict):
            convertedDict = {}
            for key, value in inputDict.items():
                try:
                    # Try to convert to an integer or float
                    if '.' in str(value):
                        convertedDict[key] = float(value)
                    else:
                        convertedDict[key] = int(value)
                except (ValueError, TypeError):
                    # Keep the original value if it cannot be converted
                    convertedDict[key] = value
            return convertedDict

        metadata = convertValuesToNumeric(metadata)
        metadata["imageTimestamp"] = pd.to_datetime(metadata['XMP:CreateDate'], format="%Y:%m:%d %H:%M:%S%z")
        
        # Load home location
        parentFolder = os.path.dirname(imagePath)
        csvFiles = [f for f in os.listdir(parentFolder) if f.endswith('.csv')]
        if not csvFiles:
            raise FileNotFoundError("No telemetry CSV file found in the parent folder.")
        telemPath = os.path.join(parentFolder, csvFiles[0])
        telem = pd.read_csv(telemPath, index_col=0, parse_dates=[0], low_memory=False, skiprows=1, nrows=10)
        try:
            metadata["homeLat"] = float(telem.iloc[-1]['HOME.latitude'])
            metadata["homeLon"] = float(telem.iloc[-1]['HOME.longitude'])
        except:
            raise ValueError(f"Invalid home location in telemetry CSV file. Lat: {telem.iloc[-1]['HOME.latitude']}, Lon: {telem.iloc[-1]['HOME.longitude']}")
        
        param = ImageParam(metadata)
        return DroneImage(image, param)
    
    def getDroneVideoFrame(self, videoPath : str, timestamp : Timestamp, offset : int) -> DroneImage:
        with ExifToolHelper() as et:
            metadataFile = et.get_metadata(videoPath)[0]

        # Load metadata
        metadataPath = videoPath.replace(".MP4", "_videoTelem.csv")
        metadataAll = pd.read_csv(metadataPath, index_col=0, parse_dates=[0], low_memory=False)

        # Find the frame number for the given timestamp
        if offset < 0:
            timestamp -= Timedelta(f"0:0:{-offset}")
        else:
            timestamp += Timedelta(f"0:0:{offset}")

        metadata = metadataAll.iloc[metadataAll.index.get_indexer([timestamp], method='nearest')].to_dict(orient='records')[0]
        frameNumber = int(metadata['frameIdx'])
        metadata["imageTimestamp"] = timestamp
        metadata = metadata | metadataFile

        # Extract the frame
        cap = cv2.VideoCapture(videoPath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"WARN: Failed to extract frame {frameNumber} from video {videoPath}, storing empty frame.")
            capOrigin = cv2.VideoCapture(videoPath)
            frame = np.zeros((int(capOrigin.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capOrigin.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)

        return DroneImage(frame, ImageParam(metadata)) 
        
if __name__ == "__main__":
    testImage = "data/testSet/mainDrone/DJI_20250130164545_0012_V_11.JPG"
    testVideo = "data/testSet/refDrone/DJI_0039_zebras.MP4"
    reader = DjiImageReader()
    droneImage = reader.getDroneImage(testImage)
    timestamp = droneImage.param.timestamp + Timedelta("0:0:4")
    # timestamp = pd.to_datetime("2025:01:30 16:45:45+03:00", format="%Y:%m:%d %H:%M:%S%z")
    # timestamp = pd.to_datetime("2025:01:30 16:54:31+03:00", format="%Y:%m:%d %H:%M:%S%z")
    droneFrame = reader.getDroneVideoFrame(testVideo, timestamp)
    # droneFrame.show()
    # cv2.waitKey(0)
    print(droneFrame)
    # print(droneImage.param.metadata)
