from pathlib import Path  # noqa: I001

from dataProcessor import DataProcessor
from imageLabeller import ImageLabeller
from objectPosition import ObjectPosition
from resultPlotter import ResultPlotter
from elevationMap import Fenswood
from errorRegression import ErrorRegression


dataFolders = [
    Path("data/uk/Fenswood_SquareAllRun1_RGB"),
    Path("data/uk/Fenswood_SquareAllRun2_RGB"),
    Path("data/uk/Fenswood_SquareAllRun3_RGB"),
    Path("data/uk/Fenswood_SquareAllRun4_RGB"),
]

dataProcessors = [DataProcessor(dataFolder) for dataFolder in dataFolders]
labeller = ImageLabeller()
plotter = ResultPlotter(
    groundMapPath=Path("data/maps/orthophoto/fenswood/odm_orthoCropped.tif")
)
errorRegression = ErrorRegression()
localiser = ObjectPosition(USE_ELEVATION=False, MAP=Fenswood())

## SYNC DATASET
for dp in dataProcessors:
    mainImages = dp.loadImagesFromFolder()
    dp.saveProcessedData(
        {"mainImages": mainImages, "dataFolder": dp.dataFolder}, "synced_images"
    )


## LABEL MAIN IMAGES
# for dataFolder in dataFolders:
#     data = processor.loadLatestProcessedData(dataFolder, "synced_images")
#     mainLabelled = labeller.labelSingleImages(data['mainImages'], ["C", "Dr", "Dd", "Dl", "Rr", "Rd", "Rl", "Lr", "Ld", "Ll", "U"], showDialog=False)
#     processor.saveProcessedData({'mainLabelled': mainLabelled, 'dataFolder': data['dataFolder']}, dataFolder, "labelled_mainOnly")

# ## DISPLAY LABELLED IMAGES
for dp in dataProcessors:
    data = dp.loadLatestProcessedData("labelled_mainOnly")
    labeller.displayLabelledImages(data["mainLabelled"])
