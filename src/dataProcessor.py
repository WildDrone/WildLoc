import os # noqa: I001
import pickle
from datetime import datetime
import json 
from tqdm import tqdm
from pathlib import Path

from djiImageReader import DjiImageReader

class DataProcessor:
    def __init__(self, dataFolder: Path):
        self.dataFolder = dataFolder
        self.processedDataPath = dataFolder / "preprocessed"

    def loadImagesFromFolder(self):
        folder = self.dataFolder / "raw" / "mainDrone"
        if not os.path.exists(folder):
            raise FileNotFoundError(f"The folder {folder} does not exist.")
        reader = DjiImageReader()
        images = []
        imagePaths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.JPG')]
        imagePaths.sort()
        for imagePath in tqdm(imagePaths, desc="Loading images"):
            droneImage = reader.getDroneImage(imagePath)
            images.append(droneImage)
        return images

    def extractFramesFromVideo(self, videoPath, images, offset=0):
        reader = DjiImageReader()
        frames = []
        for image in tqdm(images, desc="Extracting frames"):
            timestamp = image.param.timestamp
            frame = reader.getDroneVideoFrame(videoPath, timestamp, offset)
            frames.append(frame)
        return frames

    def saveProcessedData(self, data : dict, prefix : str):
        processedDataFolder = self.processedDataPath
        if not os.path.exists(processedDataFolder):
            os.makedirs(processedDataFolder)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pickleFileName = os.path.join(processedDataFolder, f'{prefix}_{timestamp}.pkl')

        with open(pickleFileName, 'wb') as f:
            pickle.dump(data, f)
       
    def syncDataSet(self):
        mainDroneFolder = os.path.join(self.dataFolder / "raw", 'mainDrone')
        refDroneFolder = os.path.join(self.dataFolder / "raw", 'refDrone')
        videoPath = next(os.path.join(refDroneFolder, f) for f in os.listdir(refDroneFolder) if f.endswith('.MP4'))
        
        syncParamPath = os.path.join(self.dataFolder / "raw", "syncParam.json")
        with open(syncParamPath, "r") as f:
            data = f.read()
            syncParam = json.loads(data)  

        mainImages = self.loadImagesFromFolder(mainDroneFolder)
        refImages = self.extractFramesFromVideo(videoPath, mainImages, offset=syncParam["droneOffset"])
        return mainImages, refImages

    def loadLatestProcessedData(self, prefix : str):
        processedDataFolder = self.processedDataPath
        pickleFiles = [f for f in os.listdir(processedDataFolder) if f.startswith(prefix) and f.endswith('.pkl')]
        pickleFiles.sort()
        latestPickleFile = pickleFiles[-1]
        pickleFileName = os.path.join(processedDataFolder, latestPickleFile)
        with open(pickleFileName, 'rb') as f:
            data = pickle.load(f)
        return data