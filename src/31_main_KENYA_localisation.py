import pandas as pd # noqa: I001
from pathlib import Path

from dataProcessor import DataProcessor
from resultPlotter import ResultPlotter
from errorRegression import ErrorRegression
from relativePosition import RelativePosition


if __name__ == "__main__":
    dataFolderLoc = Path("data/kenya/loc")
    dataFolderSize = Path("data/kenya/size")
    
    processorLoc = DataProcessor(dataFolderLoc)
    processorSize = DataProcessor(dataFolderSize)
    
    plotter = ResultPlotter(groundMapPath=Path("data/maps/orthophoto/opcWatertank/odm_orthophoto_lowRes.tif"))
    errorRegression = ErrorRegression()
    
    ### PROCESS LOCALISATION DATA
    data = processorLoc.loadLatestProcessedData("localised_images")
    imgsMain =  data['mainLocalised']
    imgsRef = data['refLocalised']

    errorRegression.doRegression(imgsMain, imgsRef, "Kenya")

    allData = pd.DataFrame()
    allData["DroneDistRaw"] = errorRegression.distancesDrone
    allData["DroneAltRaw"] = errorRegression.elevationsDrone
    allData["DroneDist"] = [round(x / 50) * 50 for x in errorRegression.distancesDrone] # Round to nearest 50m to achieve binning
    allData["DroneAlt"] = [round(x/5)*5  for x in errorRegression.elevationsDrone]
    allData["DroneHead"] = errorRegression.azimuthDrone
    allData["ErrDist"] = errorRegression.errorsLocalization
    allData["ErrAz"] = errorRegression.errorsAzimuth
    allData["ErrLongitudinal"] = errorRegression.errorsLongitudinal
    allData["ErrLateral"] = errorRegression.errorsLateral

    scenarioData = pd.DataFrame()
    for dist in allData["DroneDist"].unique():
        for alt in allData["DroneAlt"].unique():
            data = allData[(allData["DroneDist"] == dist) & (allData["DroneAlt"] == alt)]
            scenarioData = pd.concat([scenarioData, pd.DataFrame({
                            "DroneDist": dist,
                            "DroneAlt": alt,
                            "DroneHead": 0,
                            "MeanErrDist": data["ErrDist"].mean(),
                            "nbrSamples": len(data["ErrDist"].dropna()),
                            "MeanErrAz": data["ErrAz"].mean(),
                            "MeanErrLongitudinal": data["ErrLongitudinal"].mean(),
                            "MeanErrLateral": data["ErrLateral"].mean(),
                            "StdErrDist": data["ErrDist"].std(),
                            "StdErrAz": data["ErrAz"].std(),
                            "StdErrLongitudinal": data["ErrLongitudinal"].std(),
                            "StdErrLateral": data["ErrLateral"].std(),
                        }, index=[0])], ignore_index=True)

    scenarioData.to_csv(Path("data/kenya/loc/results/kenya_results.csv"), index=False, float_format="%.8f")

    ### PROCESS SIZE DATA
    data = processorSize.loadLatestProcessedData("localised_images")
    relativePosition = RelativePosition()
    relativePosition.getDistanceWithRefImg(data['mainLocalised'], data['refLocalised'])
    hzDistDrone, elDrone, relErrs, poiPairs, measuredDist = relativePosition.getData()

    allData = pd.DataFrame()
    allData["DroneDistRaw"] = hzDistDrone
    allData["DroneAltRaw"] = elDrone
    allData["DroneDist"] = [round(x / 50) * 50 for x in hzDistDrone]
    allData["DroneAlt"] = [round(x/5)*5  for x in elDrone]
    allData["DroneHead"] = [0] * len(elDrone)
    allData["ErrRel"] = relErrs
    allData["MeasuredDist"] = measuredDist
    # allData["POIPair"] = poiPairs


    scenarioData = pd.DataFrame()
    for dist in allData["DroneDist"].unique():
        for alt in allData["DroneAlt"].unique():
            data = allData[(allData["DroneDist"] == dist) & (allData["DroneAlt"] == alt)]
            scenarioData = pd.concat([scenarioData, pd.DataFrame({
                            "DroneDist": dist,
                            "DroneAlt": alt,
                            "DroneHead": 0,
                            "MeanErrRel": data["ErrRel"].mean(),
                            "StdErrRel": data["ErrRel"].std(),
                            "nbrSamples": len(data),
                            "MeanMeasuredDist": data["MeasuredDist"].mean(),
                            "MinMeasuredDist": data["MeasuredDist"].min(),
                            "MaxMeasuredDist": data["MeasuredDist"].max(),
                        }, index=[0])], ignore_index=True)
            

    scenarioData.to_csv(Path("data/kenya/size/results/kenya_results.csv"), index=False, float_format="%.8f")