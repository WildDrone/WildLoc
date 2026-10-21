from pathlib import Path  # noqa: I001
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np


def print_selectedResults(
    locExpData: pd.DataFrame,
    relPosExpData: pd.DataFrame,
):
    # Select the three points
    points = [
        # {"DroneDist": 0, "DroneAlt": 50},
        {"DroneDist": 150, "DroneAlt": 85},
        {"DroneDist": 400, "DroneAlt": 120},
    ]
    # Use locExpData for longitudinal/lateral errors
    for pt in points:
        row = locExpData[
            (locExpData["DroneDist"] == pt["DroneDist"])
            & (locExpData["DroneAlt"] == pt["DroneAlt"])
        ]
        rowRelPos = relPosExpData[
            (relPosExpData["DroneDist"] == pt["DroneDist"])
            & (relPosExpData["DroneAlt"] == pt["DroneAlt"])
        ]
        long_err = row["MeanErrLongitudinal"].values[0]
        lat_err = row["MeanErrLateral"].values[0]
        norm_err = np.sqrt(long_err**2 + lat_err**2)
        relErr = rowRelPos["MeanErrRel"].values[0]
        print(f"{long_err:.2f} {lat_err:.2f} {norm_err:.2f} {relErr:.2f}")

    print(
        f"Min size {relPosExpData['MinMeasuredDist'].min():.2f}m, Max size {relPosExpData['MaxMeasuredDist'].max():.2f}m, Mean size {relPosExpData['MeanMeasuredDist'].mean():.2f}m, Median nbr of samples {relPosExpData['nbrSamples'].median():.2f}"
    )
    
def plot_relError(
    scenarioData: pd.DataFrame,
    savePath: str | None = None,
    showErrorBars: bool = True,
    title: str | None = None,
):
    col = {
        50: "r",
        85: "b",
        120: "g",
    }

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    if title:
        fig.suptitle(title, fontweight="bold")

    for alt in scenarioData["DroneAlt"].unique():
        scenarioDataAlt = scenarioData[
            (scenarioData["DroneAlt"] == alt)
            & (scenarioData["DroneDist"] <= 400)
            & (scenarioData["DroneDist"] >= 100)
        ]
        color = col[alt]
        label = f"Elevation  {int(alt)}m"

        if showErrorBars:
            ax1.errorbar(
                scenarioDataAlt["DroneDist"],
                scenarioDataAlt["MeanErrRel"],
                yerr=scenarioDataAlt["StdErrRel"],
                fmt=f"{color}o-",
                ecolor=color,
                capsize=5,
                label=label,
            )
        else:
            ax1.plot(
                scenarioDataAlt["DroneDist"],
                scenarioDataAlt["MeanErrRel"],
                f"{color}o-",
                label=label,
            )

    ax1.set_xlabel("Distance from POI [m]")
    ax1.set_ylabel("Relative Error [%]")
    # ax1.set_ylim(bottom=-0.5, top=16)
    # ax1.set_title('Relative Error')
    ax1.grid(True)
    # ax1.set_xlim(left=-20, right=420)

    # Place a single legend below the plot
    ax1.legend(loc="best")
    fig.tight_layout()  # Leave space at the bottom for the legend

    if savePath:
        plt.savefig(savePath)
    else:
        plt.show()
    
def plot_absError(
    scenarioData: pd.DataFrame,
    savePath: str | None = None,
    showErrorBars: bool = True,
    title: str | None = None,
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
        scenarioDataAlt = scenarioData[
            (scenarioData["DroneAlt"] == alt) & (scenarioData["DroneDist"] <= 400)
        ]
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
            _ = ax2.errorbar(
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
            _ = ax2.plot(
                scenarioDataAlt["DroneDist"],
                scenarioDataAlt["MeanErrLateral"],
                f"{color}o-",
                label=label,
            )

        if showErrorBars:
            handles.append(h1.lines[0])
            labels.append(label)
        else:
            handles.append(h1[0])
            labels.append(label)

    ax1.set_ylabel("Longitudinal Error [m]")
    ax1.grid(True)
    ax1.set_title("Longitudinal Error")

    ax2.set_xlabel("Distance from POI [m]")
    ax2.set_ylabel("Lateral Error [m]")
    ax2.set_title("Lateral Error")
    ax2.set_ylim(bottom=-0.5, top=15)
    # ax2.set_xlim(left=-20, right=420)
    ax2.grid(True)

    # Hide x labels for ax1
    plt.setp(ax1.get_xticklabels(), visible=False)

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

if __name__ == "__main__":
    experimentData_Field = pd.read_csv(Path("data/kenya/loc/results/kenya_results.csv"))
    experimentDataRel_Field = pd.read_csv(Path("data/kenya/size/results/kenya_results.csv"))

    ##################################################
    ###### PLOT KENYA EXPERIMENTS ####################
    ##################################################    
    print("Plotting experimental data Kenya...")
    experimentData_Field = experimentData_Field[experimentData_Field["nbrSamples"] >= 5]
    experimentData_Field = experimentData_Field.sort_values(by=["DroneDist", "DroneAlt"]).reset_index(drop=True)
    plot_absError(experimentData_Field, showErrorBars=True, savePath=Path("figures/fig10-KENYA_results_loc/fig10-KENYA_results_loc.pdf"))

    experimentDataRel_Field = experimentDataRel_Field.sort_values(by=["DroneDist", "DroneAlt"]).reset_index(drop=True)
    plot_relError(experimentDataRel_Field, showErrorBars=True, savePath=Path("figures/fig11-KENYA_results_size/fig11-KENYA_results_size.pdf"))
    print_selectedResults(experimentData_Field, experimentDataRel_Field)
