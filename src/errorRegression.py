from scipy.optimize import curve_fit  # noqa: I001
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dataStructure import geoLoc, LocalisedDroneImage, LocalisedObject, Object


class ErrorRegression:
    def __init__(self):
        pass

    def doRegression(
        self, main: list[LocalisedDroneImage], ref: list[LocalisedDroneImage], name: str
    ):
        self.name = name
        if len(main) != len(ref):
            raise ValueError("Length of groundTruth and data must be the same")

        self.main, self.ref = LocalisedDroneImage.getMatchedFlatImages(main, ref)

        self.computeDistances()
        self.computeLateralError()

    def doRegressionWithSingleGroundTruth(
        self, main: list[LocalisedDroneImage], ref: dict[int, geoLoc], name: str
    ):
        self.name = name
        refImages = []
        refObjects = {}
        for id, loc in ref.items():
            refObjects[id] = LocalisedObject(Object(id, -1, -1), loc)
        for m in main:
            refImages.append(LocalisedDroneImage(m, refObjects))

        self.doRegression(main, refImages, name)

    def computeDistances(self):
        self.distancesDrone = []
        self.diagDistancesDrone = []
        self.elevationsDrone = []
        self.azimuthDrone = []
        self.errorsLongitudinal = []
        self.errorsLateral = []
        self.errorsLocalization = []
        self.errorsAzimuth = []

        for m, r in zip(self.main, self.ref):
            droneLoc = geoLoc(
                m.param.GPSLatitude, m.param.GPSLongitude, m.param.GPSAltitudeRelative
            )
            # TODO: change to take into account poi elevation
            mainPoiLoc = next(iter(m.objects.values())).loc
            refPoiLoc = next(iter(r.objects.values())).loc

            azDrone, _, hzDistDrone = droneLoc.aer(refPoiLoc, hzOnly=True)
            _, _, diagDistDrone = droneLoc.aer(refPoiLoc, hzOnly=False)
            elDrone = droneLoc.alt
            # elDrone = droneLoc.alt - refPoiLoc.alt
            azError, _, localizationError = mainPoiLoc.aer(refPoiLoc, hzOnly=True)

            errorLongitudinal = abs(
                localizationError * np.cos(np.radians(azError - azDrone))
            )
            errorLateral = abs(
                localizationError * np.sin(np.radians(np.abs(azError - azDrone)))
            )

            self.distancesDrone.append(hzDistDrone)
            self.diagDistancesDrone.append(diagDistDrone)
            self.elevationsDrone.append(elDrone)
            self.azimuthDrone.append(azDrone)
            self.errorsLongitudinal.append(errorLongitudinal)
            self.errorsLateral.append(errorLateral)
            self.errorsLocalization.append(localizationError)
            self.errorsAzimuth.append(azError)

    def plotAngleErrors(self):
        deltaH = np.array(self.errorsLongitudinal)
        h = np.array(self.distancesDrone)
        d = np.array(self.elevationsDrone)
        deltaTheta = np.degrees(np.arctan(d / ((h**2 + d**2) / deltaH - h)))

        fig, ax = plt.subplots()
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            self.distancesDrone,
            self.elevationsDrone,
            deltaTheta,
            c=deltaTheta,
            cmap="viridis",
            label="Angle Error",
        )
        fig.colorbar(scatter, ax=ax, label="Angle Error (deg)")
        ax.set_zlabel("Angle Error (deg)")
        ax.set_xlabel("Horizontal Distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.legend()
        ax.grid(True)
        plt.show()

    def fitSurface(self):

        def surfaceFunc(X, e):
            x, y = X
            return (x**2 + y**2) / (x + y / e)

        X = np.array([self.distancesDrone, self.elevationsDrone])
        self.popt, self.pcov = curve_fit(surfaceFunc, X, self.errorsLongitudinal)
        self.e = self.popt[0]
        self.eStd = np.sqrt(np.diag(self.pcov))[0]
        self.residuals = self.errorsLongitudinal - surfaceFunc(X, *self.popt)
        self.ssRes = np.sum(self.residuals**2)
        self.ssTot = np.sum(
            (self.errorsLongitudinal - np.mean(self.errorsLongitudinal)) ** 2
        )
        self.rSquared = 1 - (self.ssRes / self.ssTot)
        self.pitchError = np.degrees(np.arctan(self.e))
        self.pitchErrorStd = np.degrees(np.arctan(self.eStd))

    def fitSurfaceExplicitData(self, distDrone, elevDrone, errors, name):
        self.name = name
        self.distancesDrone = distDrone
        self.elevationsDrone = elevDrone
        self.errorsLongitudinal = errors
        self.fitSurface()

    def plotLongitudinalError(self, savePath=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            self.distancesDrone,
            self.elevationsDrone,
            self.errorsLongitudinal,
            c=self.residuals,
            cmap="viridis",
            label="Data",
        )
        fig.colorbar(scatter, ax=ax, label="Residuals")

        X, Y = np.meshgrid(
            np.linspace(min(self.distancesDrone), max(self.distancesDrone), 30),
            np.linspace(min(self.elevationsDrone), max(self.elevationsDrone), 30),
        )
        Z = (X**2 + Y**2) / (X + Y / self.e)
        ax.plot_surface(
            X,
            Y,
            Z,
            alpha=0.5,
            rstride=100,
            cstride=100,
            color="r",
            label="Fitted Surface",
        )

        ax.set_xlabel("Horizontal Distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_zlabel("Longitudinal Localization Error (m)")
        ax.legend()
        ax.view_init(elev=10, azim=140)  # Set the camera angle
        plt.tight_layout()

        if savePath:
            plt.savefig(savePath)
        else:
            plt.show()

    def computeLateralError(self):
        self.headingError = np.degrees(
            np.arctan(np.array(self.errorsLateral) / np.array(self.diagDistancesDrone))
        )
        self.meanHeadingError = np.mean(self.headingError)
        self.stdHeadingError = np.std(self.headingError)

        self.residualsLat = self.errorsLateral - np.array(
            self.diagDistancesDrone
        ) * np.tan(np.radians(np.mean(self.headingError)))
        self.ssResLat = np.sum(self.residualsLat**2)
        self.ssTotLat = np.sum((self.errorsLateral - np.mean(self.errorsLateral)) ** 2)
        self.rSquaredLat = 1 - (self.ssResLat / self.ssTotLat)

    def plotLateralError(self, savePath=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.diagDistancesDrone, self.errorsLateral, label="Data")

        x = np.linspace(min(self.diagDistancesDrone), max(self.diagDistancesDrone), 100)
        y = np.tan(np.radians(np.mean(self.headingError))) * x
        ax.plot(x, y, color="r", label="Fitted Line")

        ax.set_xlabel("Diagonal Distance (m)")
        ax.set_ylabel("Heading Error (deg)")
        ax.legend()
        ax.grid(True)

        if savePath:
            plt.savefig(savePath)
        else:
            plt.show()

    def getErrorTable(self, includeHeading=True):
        if includeHeading:
            errorTable = {
                "RUN": self.name,
                "Pitch Error [°]": np.degrees(np.arctan(self.e)),
                "Pitch Error Std [°]": np.degrees(np.arctan(self.eStd)),
                "Pitch Error R²": self.rSquared,
                "Heading Error [°]": self.meanHeadingError,
                "Heading Error Std [°]": self.stdHeadingError,
                "Heading Error R²": self.rSquaredLat,
            }
        else:
            errorTable = {
                "RUN": self.name,
                "Pitch Error [°]": np.degrees(np.arctan(self.e)),
                "Pitch Error Std [°]": np.degrees(np.arctan(self.eStd)),
                "Pitch Error R²": self.rSquared,
            }
        return pd.DataFrame(errorTable, index=[0])


if __name__ == "__main__":
    pass
