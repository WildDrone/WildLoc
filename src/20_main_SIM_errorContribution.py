from pathlib import Path  # noqa: I001
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from objectPosition import ObjectPosition
from camera import CamM3Tz


class NoiseProperties:
    def __init__(self):
        self.x_image_bias = 0.0
        self.x_image_std = 0.0
        self.y_image_bias = 0.0
        self.y_image_std = 0.0
        self.f_bias = 0.0
        self.f_std = 0.0
        self.cx_bias = 0.0
        self.cx_std = 0.0
        self.cy_bias = 0.0
        self.cy_std = 0.0
        self.gb_pitch_bias = 0.0
        self.gb_pitch_std = 0.0
        self.gb_yaw_bias = 0.0
        self.gb_yaw_std = 0.0
        self.northing_drone_bias = 0.0
        self.northing_drone_std = 0.0
        self.easting_drone_bias = 0.0
        self.easting_drone_std = 0.0
        self.alt_drone_bias = 0.0
        self.alt_drone_std = 0.0

    def loadFromRepresentation(self, rep: str):
        # Parse the representation string to extract noise properties
        rep = rep.removeprefix("NoiseProps(").removesuffix(")")
        properties = rep.split(", ")
        key_mapping = {
            "x_img_b": "x_image_bias",
            "x_img_s": "x_image_std",
            "y_img_b": "y_image_bias",
            "y_img_s": "y_image_std",
            "f_b": "f_bias",
            "f_s": "f_std",
            "cx_b": "cx_bias",
            "cx_s": "cx_std",
            "cy_b": "cy_bias",
            "cy_s": "cy_std",
            "gb_p_b": "gb_pitch_bias",
            "gb_p_s": "gb_pitch_std",
            "gb_y_b": "gb_yaw_bias",
            "gb_y_s": "gb_yaw_std",
            "n_d_b": "northing_drone_bias",
            "n_d_s": "northing_drone_std",
            "e_d_b": "easting_drone_bias",
            "e_d_s": "easting_drone_std",
            "alt_d_b": "alt_drone_bias",
            "alt_d_s": "alt_drone_std",
        }
        for prop in properties:
            key, value = prop.split("=")
            if key in key_mapping:
                setattr(self, key_mapping[key], float(value))
        return self

    def print(self):
        for key, value in self.__dict__.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")

    def __repr__(self):
        return f"NP(x_img_b={self.x_image_bias:.1f}, x_img_s={self.x_image_std:.1f}, y_img_b={self.y_image_bias:.1f}, y_img_s={self.y_image_std:.1f}, f_b={self.f_bias:.1f}, f_s={self.f_std:.1f}, cx_b={self.cx_bias:.1f}, cx_s={self.cx_std:.1f}, cy_b={self.cy_bias:.1f}, cy_s={self.cy_std:.1f}, gb_p_b={self.gb_pitch_bias:.1f}, gb_p_s={self.gb_pitch_std:.1f}, gb_y_b={self.gb_yaw_bias:.1f}, gb_y_s={self.gb_yaw_std:.1f}, n_d_b={self.northing_drone_bias:.1f}, n_d_s={self.northing_drone_std:.1f}, e_d_b={self.easting_drone_bias:.1f}, e_d_s={self.easting_drone_std:.1f}, alt_d_b={self.alt_drone_bias:.1f}, alt_d_s={self.alt_drone_std:.1f})"


class OperatingPoint:
    def __init__(
        self,
        x_image,
        y_image,
        f,
        cx,
        cy,
        gb_pitch,
        gb_yaw,
        northing_drone,
        easting_drone,
        alt_drone,
        refHead,
        refAlt,
        refDist,
    ):
        self.x_image = x_image
        self.y_image = y_image
        self.f = f
        self.cx = cx
        self.cy = cy
        self.gb_pitch = gb_pitch
        self.gb_yaw = gb_yaw
        self.northing_drone = northing_drone
        self.easting_drone = easting_drone
        self.alt_drone = alt_drone
        self.refHead = refHead
        self.refAlt = refAlt
        self.refDist = refDist

    def get(self):
        return [
            self.x_image,
            self.y_image,
            self.f,
            self.cx,
            self.cy,
            self.gb_pitch,
            self.gb_yaw,
            self.easting_drone,
            self.northing_drone,
            self.alt_drone,
        ]

    def asDict(self):
        return {
            "x_image": self.x_image,
            "y_image": self.y_image,
            "f": self.f,
            "cx": self.cx,
            "cy": self.cy,
            "gb_pitch": self.gb_pitch,
            "gb_yaw": self.gb_yaw,
            "northing_drone": self.northing_drone,
            "easting_drone": self.easting_drone,
            "alt_drone": self.alt_drone,
            "refHead": self.refHead,
            "refAlt": self.refAlt,
            "refDist": self.refDist,
        }

    def getWithNoise(
        self, noiseProp: NoiseProperties, noiseRealisation: list[float] | None = None
    ) -> tuple[list[float], list[float]]:
        if noiseRealisation is None:
            noise = [
                np.random.normal(noiseProp.x_image_bias, noiseProp.x_image_std),
                np.random.normal(noiseProp.y_image_bias, noiseProp.y_image_std),
                np.random.normal(noiseProp.f_bias, noiseProp.f_std),
                np.random.normal(noiseProp.cx_bias, noiseProp.cx_std),
                np.random.normal(noiseProp.cy_bias, noiseProp.cy_std),
                np.random.normal(noiseProp.gb_pitch_bias, noiseProp.gb_pitch_std),
                np.random.normal(noiseProp.gb_yaw_bias, noiseProp.gb_yaw_std),
                np.random.normal(
                    noiseProp.easting_drone_bias, noiseProp.easting_drone_std
                ),
                np.random.normal(
                    noiseProp.northing_drone_bias, noiseProp.northing_drone_std
                ),
                np.random.normal(noiseProp.alt_drone_bias, noiseProp.alt_drone_std),
            ]
        else:
            noise = noiseRealisation
            # Take a fresh noise realisation for image noise
            # noise[0] = np.random.normal(noiseProp.x_image_bias, noiseProp.x_image_std)
            # noise[1] = np.random.normal(noiseProp.y_image_bias, noiseProp.y_image_std)

        noisyOp = (np.array(self.get()) + np.array(noise)).tolist()
        return noisyOp, noise


class ErrorContributionAnalysis:
    def __init__(self, samplePerOp=50):
        self.objPos = ObjectPosition(USE_ELEVATION=False)
        self.cam = CamM3Tz()
        self.samplePerOp = samplePerOp

    def computeOperatingPoints(
        self,
        setHeading=[0],
        setAltRel=[50, 85, 120],
        setDist=[0, 50, 100, 150, 200, 250, 300, 350, 400],
    ) -> list[OperatingPoint]:
        sind = lambda degrees: math.sin(math.radians(degrees))
        cosd = lambda degrees: math.cos(math.radians(degrees))
        operatingPoints = []

        for heading in setHeading:
            for alt in setAltRel:
                for dist in setDist:
                    operatingPoints.append(
                        OperatingPoint(
                            self.cam.getCx(),
                            self.cam.getCy(),
                            self.cam.getFocalLength(),
                            self.cam.getCx(),
                            self.cam.getCy(),
                            np.degrees(math.atan2(-alt, dist)),
                            heading,
                            dist * cosd(heading),
                            dist * sind(heading),
                            alt,
                            heading,
                            alt,
                            dist,
                        )
                    )
        return operatingPoints

    def computeOperatingPointsSecondObject(
        self, ops: list[OperatingPoint], objOffsetNorth: float, objOffsetEast: float
    ) -> list[OperatingPoint]:
        # Compute the intersection of the ray from the drone to the second object with the image plane
        # The second object is offset from the first object by a given distance laterally
        # Assuming heading is 0
        opsSecondObject = []
        secondObjectPos = np.array([[objOffsetNorth], [objOffsetEast], [0]])
        for op in ops:
            C_cw = R.from_euler(
                "ZY", [op.gb_yaw, op.gb_pitch], degrees=True
            ).as_matrix()  # Rotation matrix from camera frame to world frame
            cameraPos_w = np.array(
                [[op.northing_drone], [op.easting_drone], [-op.alt_drone]]
            )
            cameraPos_c = np.array([[0], [0], [0]])
            normalToImagePlane_c = np.array([[1], [0], [0]])
            pointOnImagePlane_c = np.array([[op.f], [0], [0]])

            # Compute the intersection of the ray from the drone to the second object with the image plane
            # The ray is defined by the camera position and the second object position
            rayDirection_w = secondObjectPos - cameraPos_w
            rayDirection_w = rayDirection_w / np.linalg.norm(
                rayDirection_w
            )  # Normalize the direction vector
            rayDirection_c = np.matmul(
                C_cw, rayDirection_w
            )  # TODO: this works, but why is it not C_cw.T?
            t = np.dot(
                (pointOnImagePlane_c - cameraPos_c).ravel(),
                normalToImagePlane_c.ravel(),
            ) / np.dot(rayDirection_c.ravel(), normalToImagePlane_c.ravel())
            intersectionPoint_c = cameraPos_c + t * rayDirection_c

            opSecondObject = OperatingPoint(
                x_image=intersectionPoint_c[1, 0] + op.cx,
                y_image=-intersectionPoint_c[2, 0]
                + op.cy,  # TODO: this works, but why is minus here (related to C_cw todo above?)
                f=op.f,
                cx=op.cx,
                cy=op.cy,
                gb_pitch=op.gb_pitch,
                gb_yaw=op.gb_yaw,
                northing_drone=op.northing_drone,
                easting_drone=op.easting_drone,
                alt_drone=op.alt_drone,
                refHead=op.refHead,
                refAlt=op.refAlt,
                refDist=op.refDist,
            )
            opsSecondObject.append(opSecondObject)
        return opsSecondObject

    def computeScenario(self, ops: list[OperatingPoint], noise: NoiseProperties):
        scenarioData = pd.DataFrame(
            np.zeros((len(ops), 13)),
            columns=[
                "DroneDist",
                "DroneAlt",
                "DroneHead",
                "MeanErrDist",
                "MeanErrAz",
                "MeanErrLongitudinal",
                "MeanErrLateral",
                "StdErrDist",
                "StdErrAz",
                "StdErrLongitudinal",
                "StdErrLateral",
                "poiEasting",
                "poiNorthing",
            ],
        )
        for j, op in enumerate(ops):
            opData = pd.DataFrame(
                np.zeros((self.samplePerOp, 4)),
                columns=["ErrDist", "ErrAz", "ErrLongitudinal", "ErrLateral"],
            )

            for i in range(self.samplePerOp):
                opWithNoise, nr = op.getWithNoise(noise)
                poiLoc = self.objPos.getObjectPosition(*opWithNoise)

                poiNorthing = poiLoc["northingFromDrone"] - opWithNoise[8]
                poiEasting = poiLoc["eastingFromDrone"] - opWithNoise[7]
                distErr = np.linalg.norm([poiNorthing, poiEasting])
                azErr = math.degrees(math.atan2(poiEasting, poiNorthing))
                errLongitudinal = distErr * np.cos(np.radians(azErr - op.refHead))
                errLateral = distErr * np.sin(np.radians(azErr - op.refHead))

                opData.loc[i] = [distErr, azErr, errLongitudinal, errLateral]

            scenarioData.loc[j] = [
                op.refDist,
                op.refAlt,
                op.refHead,
                opData["ErrDist"].mean(),
                opData["ErrAz"].mean(),
                opData["ErrLongitudinal"].mean(),
                opData["ErrLateral"].mean(),
                opData["ErrDist"].std(),
                opData["ErrAz"].std(),
                opData["ErrLongitudinal"].std(),
                opData["ErrLateral"].std(),
                poiEasting,
                poiNorthing,
            ]

        return scenarioData

    def computeScenarioTwoObjects(
        self,
        ops: list[OperatingPoint],
        opsSecObj: list[OperatingPoint],
        noise: NoiseProperties,
        objSize: float,
    ):
        scenarioDataTwoObjects = pd.DataFrame(
            np.zeros((len(ops), 7)),
            columns=[
                "DroneDist",
                "DroneAlt",
                "DroneHead",
                "MeanDist",
                "MeanErrRel",
                "StdDist",
                "StdErrRel",
            ],
        )

        for j, (op, opSecObj) in enumerate(zip(ops, opsSecObj)):
            opData = pd.DataFrame(
                np.zeros((self.samplePerOp, 2)), columns=["Dist", "RelErr"]
            )

            for i in range(self.samplePerOp):
                opWithNoise, noiseRealisation = op.getWithNoise(noise)
                poiLoc = self.objPos.getObjectPosition(*opWithNoise)
                opSecObjWithNoise, _ = opSecObj.getWithNoise(noise, noiseRealisation)
                poiLocSecObj = self.objPos.getObjectPosition(*opSecObjWithNoise)

                dist = np.linalg.norm(
                    [
                        poiLocSecObj["northingFromDrone"] - poiLoc["northingFromDrone"],
                        poiLocSecObj["eastingFromDrone"] - poiLoc["eastingFromDrone"],
                    ]
                )
                relErr = np.abs(dist - objSize) / objSize * 100

                opData.loc[i] = [dist, relErr]

            scenarioDataTwoObjects.loc[j] = [
                op.refDist,
                op.refAlt,
                op.refHead,
                opData["Dist"].mean(),
                opData["RelErr"].mean(),
                opData["Dist"].std(),
                opData["RelErr"].std(),
            ]

        return scenarioDataTwoObjects

    def plot_sim_absError(
        self,
        scenarioData: pd.DataFrame,
        savePath: str = None,
        showErrorBars: bool = True,
        title: str = None,
    ):
        col = {
            50: "r",
            85: "b",
            120: "g",
        }

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
        if title:
            fig.suptitle(title, fontweight="bold")

        handles = []
        labels = []
        for alt in scenarioData["DroneAlt"].unique():
            scenarioDataAlt = scenarioData[scenarioData["DroneAlt"] == alt]
            color = col[alt]
            label = f"Elevation  {int(alt)}m"
            if showErrorBars:
                h1 = ax1.errorbar(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrLongitudinal"],
                    yerr=scenarioDataAlt["StdErrLongitudinal"],
                    fmt=f"{color}o-",
                    ecolor=color,
                    capsize=5,
                    label=label,
                )
                h2 = ax2.errorbar(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrLateral"],
                    yerr=scenarioDataAlt["StdErrLateral"],
                    fmt=f"{color}o-",
                    ecolor=color,
                    capsize=5,
                    label=label,
                )
            else:
                h1 = ax1.plot(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrLongitudinal"],
                    f"{color}o-",
                    label=label,
                )
                h2 = ax2.plot(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrLateral"],
                    f"{color}o-",
                    label=label,
                )
            if showErrorBars:
                handles.append(h1.lines[0])
            else:
                handles.append(h1[0])
            labels.append(label)

        ax1.set_ylabel("Longitudinal Error [m]")
        ax1.set_ylim(bottom=-0.5, top=15)
        ax1.set_title("Longitudinal Error")
        ax1.grid(True)

        ax2.set_xlabel("Distance from POI [m]")
        ax2.set_ylabel("Lateral Error [m]")
        ax2.set_title("Lateral Error")
        ax2.set_ylim(bottom=-0.5, top=15)
        ax2.grid(True)

        # Hide x labels for ax1
        plt.setp(ax1.get_xticklabels(), visible=False)

        # Place a single legend below the plots
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.55, 0.01),
        )
        fig.tight_layout(
            rect=[0, 0.04, 1, 1]
        )  # Leave space at the bottom for the legend

        if savePath:
            plt.savefig(savePath)
        else:
            plt.show()

    def plot_sim_absErrMaxAndInter(
        self,
        worstCaseScenarioData: pd.DataFrame,
        scenarioData: pd.DataFrame,
        savePath: str = None,
        title: str = None,
    ):
        col = {
            50: "r",
            85: "b",
            120: "g",
        }

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
        if title:
            fig.suptitle(title, fontweight="bold")

        # LONGITUDINAL
        # Compute max evenvelope
        max_longitudinal = worstCaseScenarioData.groupby("DroneDist")[
            "MeanErrLongitudinal"
        ].apply(lambda x: np.abs(x).max())
        stderr_longitudinal = worstCaseScenarioData.groupby("DroneDist")[
            "StdErrLongitudinal"
        ].apply(lambda x: np.abs(x).max())
        # Create shaded area
        ax1.fill_between(
            max_longitudinal.index,
            -max_longitudinal.values - stderr_longitudinal.values,
            max_longitudinal.values + stderr_longitudinal.values,
            color="grey",
            alpha=0.2,
            label="Max Error Region",
        )
        ax1.fill_between(
            max_longitudinal.index,
            -max_longitudinal.values,
            max_longitudinal.values,
            color="grey",
            alpha=0.2,
            label="Max Error Region",
        )
        # Plot error bar (manual cap drawing to have only one side)
        ax1.errorbar(
            max_longitudinal.index,
            max_longitudinal.values,
            yerr=[
                np.zeros_like(stderr_longitudinal.values),
                stderr_longitudinal.values,
            ],
            color="grey",
            linestyle="--",
            label="Max Longitudinal Error",
            capsize=0,
        )
        for x, y, err in zip(
            max_longitudinal.index, max_longitudinal.values, stderr_longitudinal.values
        ):
            ax1.plot([x - 5, x + 5], [y + err, y + err], color="grey", linewidth=1)
        ax1.errorbar(
            max_longitudinal.index,
            -max_longitudinal.values,
            yerr=[
                stderr_longitudinal.values,
                np.zeros_like(stderr_longitudinal.values),
            ],
            color="grey",
            linestyle="--",
            label="Min Longitudinal Error",
            capsize=0,
        )
        for x, y, err in zip(
            max_longitudinal.index, -max_longitudinal.values, stderr_longitudinal.values
        ):
            ax1.plot([x - 5, x + 5], [y - err, y - err], color="grey", linewidth=1)

        # LATERAL
        # Compute max envelope
        max_lateral = worstCaseScenarioData.groupby("DroneDist")[
            "MeanErrLateral"
        ].apply(lambda x: np.abs(x).max())
        stderr_lateral = worstCaseScenarioData.groupby("DroneDist")[
            "StdErrLateral"
        ].apply(lambda x: np.abs(x).max())
        # Create shaded area
        ax2.fill_between(
            max_lateral.index,
            -max_lateral.values - stderr_lateral.values,
            max_lateral.values + stderr_lateral.values,
            color="grey",
            alpha=0.2,
            label="Max Error Region",
        )
        ax2.fill_between(
            max_lateral.index,
            -max_lateral.values,
            max_lateral.values,
            color="grey",
            alpha=0.2,
            label="Max Error Region",
        )
        # Plot error bar (manual cap drawing to have only one side)
        ax2.errorbar(
            max_lateral.index,
            max_lateral.values,
            yerr=[np.zeros_like(stderr_lateral.values), stderr_lateral.values],
            color="grey",
            linestyle="--",
            label="Max Lateral Error",
            capsize=0,
        )
        for x, y, err in zip(
            max_lateral.index, max_lateral.values, stderr_lateral.values
        ):
            ax2.plot([x - 5, x + 5], [y + err, y + err], color="grey", linewidth=1)
        ax2.errorbar(
            max_lateral.index,
            -max_lateral.values,
            yerr=[stderr_lateral.values, np.zeros_like(stderr_lateral.values)],
            color="grey",
            linestyle="--",
            label="Min Lateral Error",
            capsize=0,
        )
        for x, y, err in zip(
            max_lateral.index, -max_lateral.values, stderr_lateral.values
        ):
            ax2.plot([x - 5, x + 5], [y - err, y - err], color="grey", linewidth=1)

        handles = []
        labels = []
        for alt in scenarioData["DroneAlt"].unique():
            scenarioDataAlt = scenarioData[scenarioData["DroneAlt"] == alt]
            color = col[alt]
            label = f"Elevation  {int(alt)}m"
            h1 = ax1.errorbar(
                scenarioDataAlt["DroneDist"],
                scenarioDataAlt["MeanErrLongitudinal"],
                yerr=scenarioDataAlt["StdErrLongitudinal"],
                fmt=f"{color}o-",
                ecolor=color,
                capsize=5,
                label=label,
            )
            h2 = ax2.errorbar(
                scenarioDataAlt["DroneDist"],
                scenarioDataAlt["MeanErrLateral"],
                yerr=scenarioDataAlt["StdErrLateral"],
                fmt=f"{color}o-",
                ecolor=color,
                capsize=5,
                label=label,
            )

            # Only collect handles/labels once (from ax1)
            handles.append(h1[0])
            labels.append(label)

        ax1.set_ylabel("Longitudinal Error [m]")
        ax1.set_title("Longitudinal Error")
        ax1.set_ylim(bottom=-27, top=27)
        ax1.grid(True)

        ax2.set_xlabel("Distance from POI [m]")
        ax2.set_ylabel("Lateral Error [m]")
        ax2.set_title("Lateral Error")
        ax2.set_ylim(bottom=-27, top=27)
        ax2.grid(True)

        # Place a single legend below the plots, further down to avoid overlap
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.55, 0.01),
        )
        fig.tight_layout(
            rect=[0, 0.04, 1, 1]
        )  # Leave space at the bottom for the legend

        if savePath:
            plt.savefig(savePath)
        else:
            plt.show()

    def plot_sim_relErrorMax(
        self,
        scenarioDataTwoObjectsLat: pd.DataFrame,
        scenarioDataTwoObjectsLon: pd.DataFrame,
        savePath: str = None,
        showErrorBars: bool = True,
        title: str = None,
    ):
        col = {
            50: "r",
            85: "b",
            120: "g",
        }

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
        if title:
            fig.suptitle(title, fontweight="bold")

        handles = []
        labels = []
        for alt in scenarioDataTwoObjectsLon["DroneAlt"].unique():
            scenarioDataAlt = scenarioDataTwoObjectsLon[
                scenarioDataTwoObjectsLon["DroneAlt"] == alt
            ]
            color = col[alt]
            label = f"Elevation  {int(alt)}m"
            if alt == 50:
                # Shaded area from (MeanErrRel + StdErrRel) to 0
                ax1.fill_between(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrRel"] + scenarioDataAlt["StdErrRel"],
                    0,
                    color="grey",
                    alpha=0.2,
                )
                # Shaded area from MeanErrRel to 0
                ax1.fill_between(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrRel"],
                    0,
                    color="grey",
                    alpha=0.25,
                )
                ax1.plot(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrRel"] + scenarioDataAlt["StdErrRel"],
                    color="grey",
                    linestyle="--",
                )
                ax1.plot((0, 400), (0, 0), color="grey", linestyle="--")

            if showErrorBars:
                h1 = ax1.errorbar(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrRel"],
                    yerr=scenarioDataAlt["StdErrRel"],
                    fmt=f"{color}o-",
                    ecolor=color,
                    capsize=5,
                    label=label,
                )
            else:
                h1 = ax1.plot(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrRel"],
                    f"{color}o-",
                    label=label,
                )
            labels.append(label)

        for alt in scenarioDataTwoObjectsLat["DroneAlt"].unique():
            scenarioDataAlt = scenarioDataTwoObjectsLat[
                scenarioDataTwoObjectsLat["DroneAlt"] == alt
            ]
            color = col[alt]
            label = f"Elevation  {int(alt)}m"
            if alt == 50:
                # Shaded area from (MeanErrRel + StdErrRel) to 0
                ax2.fill_between(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrRel"] + scenarioDataAlt["StdErrRel"],
                    0,
                    color="grey",
                    alpha=0.2,
                )
                # Shaded area from MeanErrRel to 0
                ax2.fill_between(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrRel"],
                    0,
                    color="grey",
                    alpha=0.25,
                )
                ax2.plot(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrRel"] + scenarioDataAlt["StdErrRel"],
                    color="grey",
                    linestyle="--",
                )
                ax2.plot((0, 400), (0, 0), color="grey", linestyle="--")
            if showErrorBars:
                h1 = ax2.errorbar(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrRel"],
                    yerr=scenarioDataAlt["StdErrRel"],
                    fmt=f"{color}o-",
                    ecolor=color,
                    capsize=5,
                    label=label,
                )
                handles.append(h1.lines[0])
            else:
                h1 = ax2.plot(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrRel"],
                    f"{color}o-",
                    label=label,
                )
                handles.append(h1[0])
            labels.append(label)

        ax1.set_ylabel("Relative Error [%]")
        # ax1.set_ylim(bottom=-0.5, top=9.5)
        ax1.set_title("Longitudinal object")
        ax1.grid(True)

        ax2.set_xlabel("Distance from POI [m]")
        ax2.set_ylabel("Relative Error [%]")
        # ax2.set_ylim(bottom=-0.5, top=9.5)
        ax2.set_title("Lateral object")
        ax2.grid(True)

        # Place a single legend below the plots, further down to avoid overlap
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.55, 0.01),
        )
        fig.tight_layout(
            rect=[0, 0.04, 1, 1]
        )  # Leave space at the bottom for the legend

        if savePath:
            plt.savefig(savePath)
        else:
            plt.show()

    def getNoiseProperties(self, type: str) -> NoiseProperties:
        if type == "hzPosition":
            noiseHz = NoiseProperties()
            noiseHz.easting_drone_bias = 1
            # noiseHz.easting_drone_std = 1
            noiseHz.northing_drone_bias = 1
            # noiseHz.northing_drone_std = 1
            return noiseHz
        elif type == "vertPosition":
            noiseVert = NoiseProperties()
            noiseVert.alt_drone_bias = 1
            # noiseVert.alt_drone_std = 1
            return noiseVert
        elif type == "attitude":
            noiseAtt = NoiseProperties()
            noiseAtt.gb_pitch_bias = 0.1
            noiseAtt.gb_pitch_std = 0.1
            noiseAtt.gb_yaw_bias = 1
            noiseAtt.gb_yaw_std = 1
            return noiseAtt
        elif type == "pitch":
            noisePitch = NoiseProperties()
            noisePitch.gb_pitch_bias = 0.1
            # noisePitch.gb_pitch_std = 0.1
            return noisePitch
        elif type == "yaw":
            noiseYaw = NoiseProperties()
            noiseYaw.gb_yaw_bias = 1
            # noiseYaw.gb_yaw_std = 1
            return noiseYaw
        elif type == "image":
            noiseImage = NoiseProperties()
            noiseImage.x_image_bias = 1
            noiseImage.x_image_std = 1
            noiseImage.y_image_bias = 1
            noiseImage.y_image_std = 1
            return noiseImage
        elif type == "imageX":
            noiseImageX = NoiseProperties()
            noiseImageX.x_image_bias = 5000
            # noiseImageX.x_image_std = 50
            return noiseImageX
        elif type == "imageY":
            noiseImageY = NoiseProperties()
            noiseImageY.y_image_bias = 5000
            # noiseImageY.y_image_std = 50
            return noiseImageY
        elif type == "focalLength":
            noiseFocalLength = NoiseProperties()
            noiseFocalLength.f_bias = 50
            # noiseFocalLength.gb_pitch_bias = 0.1
            # noiseFocalLength.f_std = 5
            return noiseFocalLength
        elif type == "nonTrivial":
            noiseNonTrivial = NoiseProperties()
            noiseNonTrivial.gb_pitch_bias = -0.1
            noiseNonTrivial.gb_pitch_std = 0.1
            noiseNonTrivial.alt_drone_bias = 1
            noiseNonTrivial.alt_drone_std = 1
            noiseNonTrivial.northing_drone_bias = -1
            noiseNonTrivial.northing_drone_std = 1
            noiseNonTrivial.easting_drone_bias = -1
            noiseNonTrivial.easting_drone_std = 1
            noiseNonTrivial.gb_yaw_bias = -1
            noiseNonTrivial.gb_yaw_std = 1
            noiseNonTrivial.x_image_bias = 10
            noiseNonTrivial.x_image_std = 10
            return noiseNonTrivial
        elif type == "worstCase":
            noiseWrstCase = NoiseProperties()
            noiseWrstCase.gb_pitch_bias = 0.1
            noiseWrstCase.gb_pitch_std = 0.1
            noiseWrstCase.alt_drone_bias = 1
            noiseWrstCase.alt_drone_std = 1
            noiseWrstCase.northing_drone_bias = -1
            noiseWrstCase.northing_drone_std = 1
            noiseWrstCase.y_image_bias = 10
            noiseWrstCase.y_image_std = 10

            noiseWrstCase.easting_drone_bias = -1
            noiseWrstCase.easting_drone_std = 1
            noiseWrstCase.gb_yaw_bias = 1
            noiseWrstCase.gb_yaw_std = 1
            noiseWrstCase.x_image_bias = 10
            noiseWrstCase.x_image_std = 10

            noiseWrstCase.f_bias = -40
            noiseWrstCase.f_std = 40
            return noiseWrstCase
        elif type == "debug":
            noiseTest = NoiseProperties()
            noiseTest.x_image_bias = 13
            noiseTest.y_image_bias = -13
            return noiseTest
        else:
            raise ValueError(f"Unknown noise type: {type}")


if __name__ == "__main__":
    # SETUP
    cam = CamM3Tz()
    eca = ErrorContributionAnalysis(samplePerOp=10000)
    ops = eca.computeOperatingPoints()

    ##################################################
    ###### PLOT SIMULATION EXPERIMENTS ###############
    ##################################################

    # PLOT ATTITUDE NOISE CONTRIBUTION
    print("Plotting attitude noise contribution...")
    scenarioData = eca.computeScenario(ops, eca.getNoiseProperties("attitude"))
    eca.plot_sim_absError(
        scenarioData, savePath=Path("figures/fig05-SIM_attiError_loc/fig05-SIM_attiError_loc.pdf")
    )

    # PLOT MAX ABSOLUTE ERROR SIMULATED
    print("Plotting maximum absolute error simulated...")
    scenarioData = eca.computeScenario(ops, eca.getNoiseProperties("nonTrivial"))
    worstCaseScenarioData = eca.computeScenario(
        ops, eca.getNoiseProperties("worstCase")
    )
    eca.plot_sim_absErrMaxAndInter(
        worstCaseScenarioData,
        scenarioData,
        savePath=Path("figures/fig06-SIM_worstCaseError_loc/fig06-SIM_worstCaseError_loc.pdf"),
    )

    # PLOT MAX RELATIVE ERROR SIMULATED
    print("Plotting maximum relative error simulated...")
    objectSize = 2  # in meters, used for the second object offset
    opsSecondObjectLon = eca.computeOperatingPointsSecondObject(
        ops, objOffsetNorth=objectSize, objOffsetEast=0
    )  # Second object is offset by objectSize in the north direction
    opsSecondObjectLat = eca.computeOperatingPointsSecondObject(
        ops, objOffsetNorth=0, objOffsetEast=objectSize
    )  # Second object is offset by objectSize in the east direction

    scenarioDataTwoObjectsLon = eca.computeScenarioTwoObjects(
        ops, opsSecondObjectLon, eca.getNoiseProperties("worstCase"), objectSize
    )
    scenarioDataTwoObjectsLat = eca.computeScenarioTwoObjects(
        ops, opsSecondObjectLat, eca.getNoiseProperties("worstCase"), objectSize
    )
    eca.plot_sim_relErrorMax(
        scenarioDataTwoObjectsLat,
        scenarioDataTwoObjectsLon,
        savePath=Path("figures/fig07-SIM_worstCaseError_size/fig07-SIM_worstCaseError_size.pdf"),
    )
