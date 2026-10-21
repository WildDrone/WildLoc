from pathlib import Path # noqa: I001

from dataProcessor import DataProcessor 
from imageLabeller import ImageLabeller
from objectPosition import ObjectPosition
from resultPlotter import ResultPlotter
from errorRegression import ErrorRegression
from elevationMap import OlPejetaZebraHerd2
    

if __name__ == "__main__":
    dataFolder = Path("data/kenya/loc")
    processor = DataProcessor(dataFolder)
    labeller = ImageLabeller()
    localiser = ObjectPosition(USE_ELEVATION = True, MAP = OlPejetaZebraHerd2())
    plotter = ResultPlotter(groundMapPath=Path("data/maps/orthophoto/opcWatertank/odm_orthophoto_lowRes.tif"))
    errorRegression = ErrorRegression()

    ## SYNC DATASET
    # mainImages, refImages = processor.syncDataSet()
    # processor.saveProcessedData({'mainImages': mainImages, 'refImages': refImages, 'dataFolder': dataFolder}, "synced_images")

    ## DISPLAY SYNCED IMAGES
    # data = processor.loadLatestProcessedData("synced_images")
    # labeller.displaySideBySide(data['mainImages'], data['refImages'])  
    
    # ## LABEL MAIN IMAGES
    # syncParamPath = os.path.join(dataFolder, "syncParam.json")
    # with open(syncParamPath, "r") as f:
    #     data = f.read()
    #     syncParam = json.loads(data)  
    # data = processor.loadLatestProcessedData("synced_images")
    # mainLabelled = labeller.labelSingleImages(data["mainImages"], labelsList = syncParam["labelList"], showDialog=False)
    # processor.saveProcessedData({'mainLabelled': mainLabelled, 'refImages': data['refImages'], 'dataFolder': data['dataFolder']}, "labelled_mainOnly")

    # ## REMOVE A LABEL
    # data = processor.loadLatestProcessedData("labelled_mainOnly")
    # mainLabelled = data['mainLabelled']
    # mainLabelled = labeller.removeLabel(mainLabelled, frameId=1, labelId=0)
    # labeller.displaySideBySide(mainLabelled, data['refImages'])  
    # processor.saveProcessedData({'mainLabelled': data['mainLabelled'], 'refImages': data['refImages'], 'dataFolder': data['dataFolder']}, "labelled_mainOnly")

    # ## LABEL REF SIDE BY SIDE
    # data = processor.loadLatestProcessedData("labelled_mainOnly")
    # refLabelled = labeller.labelSideBySide(data['refImages'], data["mainLabelled"])
    # processor.saveProcessedData({'mainLabelled': data["mainLabelled"], 'refLabelled': refLabelled, 'dataFolder': data['dataFolder']}, "labelled_images")

    # ## COPY LABELS FROM RGB TO THERMAL
    # dataFolder_RGB = dataFolder.replace("_T", "_RGB")
    # dataRGB = processor.loadLatestProcessedData(dataFolder_RGB, "labelled_images")
    # dataT = processor.loadLatestProcessedData("synced_images")
    # mainLabelled = labeller.copyLabels(dataRGB['mainLabelled'], dataT['mainImages'])
    # refLabelled = dataRGB['refLabelled']
    # processor.saveProcessedData({'mainLabelled': mainLabelled, 'refLabelled': refLabelled, 'dataFolder': dataT['dataFolder']}, "labelled_images")

    ## DISPLAY LABELLED IMAGES
    # data = processor.loadLatestProcessedData("labelled_images")
    # labeller.displaySideBySide(data['mainLabelled'], data['refLabelled'])

    ## LOCALISE OBJECTS
    data = processor.loadLatestProcessedData("labelled_images")
    mainLocalised = localiser.getLocalisedDroneImages(data['mainLabelled'])
    refLocalised = localiser.getLocalisedDroneImages(data['refLabelled'])
    processor.saveProcessedData({'mainLocalised': mainLocalised, 'refLocalised': refLocalised, 'dataFolder': data['dataFolder']}, "localised_images")

    ## DISPLAY LOCALISED IMAGES
    data = processor.loadLatestProcessedData("localised_images")
    plotter.displayLocalizedDroneImagesOnMap(data['mainLocalised'], data['refLocalised'], showDroneLocation=True, selectedObject=[0])
