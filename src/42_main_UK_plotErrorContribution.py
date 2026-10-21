from pathlib import Path  # noqa: I001
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

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

def print_selectedResults(
    locExpData: pd.DataFrame,
    relPosExpData: pd.DataFrame,
):
    # Select the three points
    points = [
        {"DroneDist": 0, "DroneAlt": 50},
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

        fig, (ax1b, ax1, ax2) = plt.subplots(
            3,
            1,
            figsize=(6, 7),
            gridspec_kw={"height_ratios": [0.5, 4.5, 5]},
            sharex=True,
        )
        if title:
            fig.suptitle(title, fontweight="bold")

        # Hide the spines between ax1 and ax1b
        ax1.spines["top"].set_visible(False)
        ax1b.spines["bottom"].set_visible(False)
        ax1.xaxis.tick_bottom()
        ax1b.xaxis.tick_top()
        ax1b.tick_params(labeltop=False)  # don't put tick labels at the top
        # Reduce spacing between ax1 and ax1b

        # Set ax1b to have only one tick at 20
        ax1b.set_yticks([21])
        ax1b.set_yticklabels(["21"])

        # Set y-limits for the broken axes
        ax1.set_ylim(bottom=-0.5, top=15)
        ax1b.set_ylim(20, 21.5)

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
                _ = ax1b.errorbar(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrLongitudinal"],
                    yerr=scenarioDataAlt["StdErrLongitudinal"],
                    fmt=f"{color}o-",
                    ecolor=color,
                    capsize=5,
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
                _ = ax1b.plot(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrLongitudinal"],
                    f"{color}o-",
                )
                _ = ax2.plot(
                    scenarioDataAlt["DroneDist"],
                    scenarioDataAlt["MeanErrLateral"],
                    f"{color}o-",
                    label=label,
                )
            # Only collect handles/labels once (from ax1)
            if showErrorBars:
                handles.append(h1.lines[0])
            else:
                handles.append(h1[0])
            labels.append(label)

        ax1.set_ylabel("Longitudinal Error [m]")
        ax1.grid(True)

        ax1b.set_ylabel("")
        ax1b.grid(True)
        ax1b.set_title("Longitudinal Error")

        ax2.set_xlabel("Distance from POI [m]")
        ax2.set_ylabel("Lateral Error [m]")
        ax2.set_title("Lateral Error")
        ax2.set_ylim(bottom=-0.5, top=15)
        ax2.grid(True)

        # Hide x labels for ax1 and ax1b
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1b.get_xticklabels(), visible=False)

        # Place a single legend below the plots, further down to avoid overlap
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.55, 0.01),
        )
        # Do not call tight_layout here, as it can override subplots_adjust
        fig.tight_layout(
            rect=[0, 0.04, 1, 1]
        )  # Leave space at the bottom for the legend

        # Only adjust spacing between ax1 and ax1b (the broken y-axis pair)
        # Move ax1b closer to ax1 by manually adjusting their positions
        pos1 = ax1.get_position()
        ax1.set_position([pos1.x0, pos1.y0, pos1.width, pos1.height + 0.04])

        # Draw break lines (diagonal lines) to indicate the y-axis break, with matching slopes
        d = 0.006  # size of break lines in axes coordinates
        # Horizontal break lines (short horizontal lines)
        kwargs = {"transform": ax1.transAxes, "color": "k", "clip_on": False, "linewidth": 1}
        ax1.plot([-d, d], [1, 1], **kwargs)  # top-left horizontal
        ax1.plot([1 - d, 1 + d], [1, 1], **kwargs)  # top-right horizontal

        kwargs.update(transform=ax1b.transAxes)  # switch to the broken axis
        ax1b.plot([-d, d], [0, 0], **kwargs)  # bottom-left horizontal
        ax1b.plot([1 - d, 1 + d], [0, 0], **kwargs)  # bottom-right horizontal

        if savePath:
            plt.savefig(savePath)
        else:
            plt.show()
   

if __name__ == "__main__":
    experimentData = pd.read_csv(Path("data/uk/resultsAggredated/uk_results_loc.csv"))
    experimentDataRel = pd.read_csv(Path("data/uk/resultsAggredated/uk_results_size.csv"))


    ##################################################
    ###### PLOT UK EXPERIMENTS #######################
    ##################################################
    experimentData = experimentData.sort_values(by=["DroneDist", "DroneAlt"]).reset_index(drop=True)
    plot_absError(experimentData, showErrorBars=True, savePath=Path("figures/fig13-UK_results_loc/fig13-UK_results_loc.pdf"))
    experimentDataRel = experimentDataRel.sort_values(by=["DroneDist", "DroneAlt"]).reset_index(drop=True)
    plot_relError(experimentDataRel, showErrorBars=True, savePath=Path("figures/fig14-UK_results_size/fig14-UK_results_size.pdf"))

    print_selectedResults(experimentData, experimentDataRel)
