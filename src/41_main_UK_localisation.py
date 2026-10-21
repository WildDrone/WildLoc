from pathlib import Path  # noqa: I001
import pandas as pd

from dataProcessor import DataProcessor
from imageLabeller import ImageLabeller
from objectPosition import ObjectPosition
from resultPlotter import ResultPlotter
from elevationMap import Fenswood
from errorRegression import ErrorRegression
from relativePosition import RelativePosition
from dataStructure import geoLoc


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


## LOCALISE OBJECTS
# for dp in dataProcessors:
#     data = dp.loadLatestProcessedData("labelled_mainOnly")
#     mainLocalised = localiser.getLocalisedDroneImages(data['mainLabelled'])
#     data['mainLocalised'] = mainLocalised
#     dp.saveProcessedData({'mainLocalised': mainLocalised, 'dataFolder': data['dataFolder']}, "localised_images")

## DISPLAY LOCALISED IMAGES
# for dp in dataProcessors:
#     data = dp.loadLatestProcessedData("localised_images")
#     plotter.displayLocalizedDroneImagesOnMap(data['mainLocalised'], colorIsID=True, selectedObject=[0])



imgs = []
for dp in dataProcessors:
    data = dp.loadLatestProcessedData("localised_images")
    for img in data['mainLocalised']:
        img.noImage()
    imgs += data['mainLocalised']
print("Done Loading Localised Images")

ref = {0: geoLoc(51.423445, -2.671415, 0)}
errorRegression.doRegressionWithSingleGroundTruth(
    [img.filter([0]) for img in imgs], ref, "all"
)

allData = pd.DataFrame()
allData["DroneDistRaw"] = errorRegression.distancesDrone
allData["DroneAltRaw"] = errorRegression.elevationsDrone
allData["DroneDist"] = [
    min([0, 50, 100, 150, 200, 250, 300, 350, 400], key=lambda y: abs(y - x))
    for x in errorRegression.distancesDrone
]
allData["DroneAlt"] = [
    min([50, 85, 120], key=lambda y: abs(y - x))
    for x in errorRegression.elevationsDrone
]
allData["DroneHead"] = [0] * len(errorRegression.distancesDrone)
allData["ErrDist"] = errorRegression.errorsLocalization
allData["ErrAz"] = errorRegression.errorsAzimuth
allData["ErrLongitudinal"] = errorRegression.errorsLongitudinal
allData["ErrLateral"] = errorRegression.errorsLateral

scenarioData = pd.DataFrame()
for dist in allData["DroneDist"].unique():
    for alt in allData["DroneAlt"].unique():
        data = allData[(allData["DroneDist"] == dist) & (allData["DroneAlt"] == alt)]
        scenarioData = pd.concat(
            [
                scenarioData,
                pd.DataFrame(
                    {
                        "DroneDist": dist,
                        "DroneAlt": alt,
                        "DroneHead": 0,
                        "MeanErrDist": data["ErrDist"].mean(),
                        "MeanErrAz": data["ErrAz"].mean(),
                        "MeanErrLongitudinal": data["ErrLongitudinal"].mean(),
                        "MeanErrLateral": data["ErrLateral"].mean(),
                        "StdErrDist": data["ErrDist"].std(),
                        "StdErrAz": data["ErrAz"].std(),
                        "StdErrLongitudinal": data["ErrLongitudinal"].std(),
                        "StdErrLateral": data["ErrLateral"].std(),
                        "nbrSamples": len(data),
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

scenarioData.to_csv(
    Path("data/uk/resultsAggredated/uk_results_loc.csv"),
    index=False,
    float_format="%.8f",
)

# ## COMPUTE RELATIVE POSITION
relativePosition = RelativePosition()
groundTruthMeasurements = {
    "2-4": 2.0,
    "2-9": 2.0,
    "refPos": geoLoc(51.423445, -2.671415, 0),
}

relativePosition.getDistance(imgs, groundTruthMeasurements)
hzDistDrone, elDrone, relErrs, poiPairs = relativePosition.getData()

allData = pd.DataFrame()
allData["DroneDistRaw"] = hzDistDrone
allData["DroneAltRaw"] = elDrone
allData["DroneDist"] = [
    min([0, 50, 100, 150, 200, 250, 300, 350, 400], key=lambda y: abs(y - x))
    for x in hzDistDrone
]
allData["DroneAlt"] = [min([50, 85, 120], key=lambda y: abs(y - x)) for x in elDrone]
allData["DroneHead"] = [0] * len(elDrone)
allData["ErrRel"] = relErrs
allData["POIPair"] = poiPairs


scenarioData = pd.DataFrame()
for dist in allData["DroneDist"].unique():
    for alt in allData["DroneAlt"].unique():
        data = allData[(allData["DroneDist"] == dist) & (allData["DroneAlt"] == alt)]
        scenarioData = pd.concat(
            [
                scenarioData,
                pd.DataFrame(
                    {
                        "DroneDist": dist,
                        "DroneAlt": alt,
                        "DroneHead": 0,
                        "MeanErrRel": data["ErrRel"].mean(),
                        "StdErrRel": data["ErrRel"].std(),
                        "nbrSamples": len(data),
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

scenarioData.to_csv(
    Path("data/uk/resultsAggredated/uk_results_size.csv"),
    index=False,
    float_format="%.8f",
)
