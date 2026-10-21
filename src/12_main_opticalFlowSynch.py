import cv2  # noqa: I001
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from exiftool import ExifToolHelper
import pickle
import json
from pathlib import Path
from scipy.signal import find_peaks


class OpticalFlowTracker:
    def __init__(self, video_path, start_time, end_time, selectedCorr=None):
        self.selectedCorr = selectedCorr
        self.video_path = video_path
        self.start_time = start_time
        self.end_time = end_time
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Frame count: {self.frameCount}")
        self.start_frame = int(start_time * self.fps)
        self.end_frame = int(end_time * self.fps)
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }
        self.feature_params = {
            "maxCorners": 100,
            "qualityLevel": 0.3,
            "minDistance": 7,
            "blockSize": 7,
        }
        self.tracks = []
        self.avg_speeds_x = []
        self.avg_speeds_y = []
        with ExifToolHelper() as et:
            metadata = et.get_metadata(self.video_path)[0]

        self.videoCreationTime = pd.to_datetime(
            metadata["QuickTime:CreateDate"], format="%Y:%m:%d %H:%M:%S"
        )
        self.videoCreationTime = self.videoCreationTime.tz_localize("GMT")
        print(f"Video creation time: {self.videoCreationTime}")
        self.clipStartTime = self.videoCreationTime + pd.to_timedelta(
            self.start_time, unit="s"
        )
        self.frameRate = float(metadata["QuickTime:VideoFrameRate"])

    def run(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        ret, old_frame = self.cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)
        mask = np.zeros_like(old_frame)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.video_path.with_name("opticalFlow.mp4"),
            fourcc,
            self.fps,
            (old_frame.shape[1], old_frame.shape[0]),
        )

        for frame_idx in range(self.start_frame, self.end_frame):
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **self.lk_params
            )
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            avg_speed_x = np.mean(good_new[:, 0] - good_old[:, 0])
            avg_speed_y = np.mean(good_new[:, 1] - good_old[:, 1])
            self.avg_speeds_x.append(avg_speed_x)
            self.avg_speeds_y.append(avg_speed_y)

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a = int(a)
                b = int(b)
                c = int(c)
                d = int(d)
                mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
            img = cv2.add(frame, mask)
            out.write(img)

            # cv2.imshow('frame', img)
            # if cv2.waitKey(30) & 0xFF == 27:
            #     break

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        self.cap.release()
        cv2.destroyAllWindows()
        out.release()

        self.flowSpeeds = pd.DataFrame(
            {
                "xflowSpeed_body": np.array(self.avg_speeds_y) * 2,
                "yflowSpeed_body": -np.array(self.avg_speeds_x) * 2,
                "frameIdx": range(self.start_frame, self.end_frame),
            }
        )

        # self.plotFlowSpeed()

    def plotFlowSpeed(self):
        plt.figure()
        plt.plot(
            self.flowSpeeds.index,
            self.flowSpeeds["xflowSpeed_body"],
            label="Average Speed X",
        )
        plt.plot(
            self.flowSpeeds.index,
            self.flowSpeeds["yflowSpeed_body"],
            label="Average Speed Y",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Speed")
        plt.title("Optical Flow Speeds")
        plt.legend()
        plt.show()

    def loadFlightLog(self, csv_path):
        self.flightLog = pd.read_csv(csv_path, low_memory=False, skiprows=1)
        self.flightLog["Time"] = pd.to_datetime(
            self.flightLog["CUSTOM.date [local]"]
            + " "
            + self.flightLog["CUSTOM.updateTime [local]"],
            format="%m/%d/%Y %I:%M:%S.%f %p",
        )
        self.flightLog["Time"] = self.flightLog["Time"].dt.tz_localize("GMT")
        self.flightLog = self.flightLog.set_index("Time")
        duration = self.flightLog.index[-1] - self.flightLog.index[0]
        print(f"Total flight duration: {duration}")
        # OSD.xSpeed [MPH] is east speed, OSD.ySpeed [MPH] is north speed
        self.flightLog["nSpeed_ENU_mps"] = self.flightLog["OSD.xSpeed [MPH]"] * 0.44704
        self.flightLog["eSpeed_ENU_mps"] = self.flightLog["OSD.ySpeed [MPH]"] * 0.44704
        # Body frame is x-forward, y-left, z-up
        angle = self.flightLog["OSD.yaw"]
        self.flightLog["xSpeed_body_mps"] = self.flightLog["nSpeed_ENU_mps"] * np.cos(
            np.radians(angle)
        ) + self.flightLog["eSpeed_ENU_mps"] * np.sin(np.radians(angle))
        self.flightLog["ySpeed_body_mps"] = -self.flightLog["nSpeed_ENU_mps"] * np.sin(
            np.radians(angle)
        ) + self.flightLog["eSpeed_ENU_mps"] * np.cos(np.radians(angle))
        # Image frame is x-right, y-down, z-forward
        self.flightLogSpeedOnly = self.flightLog[
            ["xSpeed_body_mps", "ySpeed_body_mps"]
        ].copy()
        # self.plotFlightLogSpeeds()

    def plotFlightLogSpeeds(self):
        plt.figure()
        plt.plot(
            self.flightLog.index,
            self.flightLog["xSpeed_body_mps"],
            label="X Speed (m/s)",
        )
        plt.plot(
            self.flightLog.index,
            self.flightLog["ySpeed_body_mps"],
            label="Y Speed (m/s)",
        )
        plt.xlabel("Time")
        plt.ylabel("Speed (m/s)")
        plt.legend()
        plt.twinx().plot(
            self.flightLog.index,
            self.flightLog["OSD.yaw"],
            color="green",
            label="Yaw (degrees)",
        )
        plt.ylabel("Yaw (degrees)")
        plt.title("Flight Log Speeds")
        plt.show()

    def computeCrossCorrelation(self):
        # Assuming flight log is 10ms and video is 30ms per frame
        # Upsample the video data to match the flight log
        # upsampledData = []
        # self.flightLog["Original"] = True
        # for row in self.flightLog.itertuples(index=False):
        #     upsampledData.append(row._asdict())
        #     for i in range(2):
        #         upsampledData.append(emptyRow)

        # self.flightLogUpsampled = pd.DataFrame(upsampledData, columns=self.flightLog.columns)
        # self.flightLogUpsampled[['xSpeed_body_mps', 'ySpeed_body_mps']] = self.flightLogUpsampled[['xSpeed_body_mps', 'ySpeed_body_mps']].interpolate()

        self.flightLogUpsampled = self.flightLog.resample(
            pd.Timedelta(1 / self.frameRate, unit="s"), origin=self.flightLog.index[0]
        ).ffill()

        # Compute cross correlation
        xCorr = np.correlate(
            self.flightLogUpsampled["xSpeed_body_mps"].values,
            self.flowSpeeds["xflowSpeed_body"].values,
            mode="valid",
        )
        yCorr = np.correlate(
            self.flightLogUpsampled["ySpeed_body_mps"].values,
            self.flowSpeeds["yflowSpeed_body"].values,
            mode="valid",
        )
        # xyCorr = np.array([np.sqrt(x * y) if x > 0 and y > 0 else 0 for x, y in zip(xCorr, yCorr)])
        xyCorr = np.correlate(
            self.flightLogUpsampled["xSpeed_body_mps"].values
            * self.flightLogUpsampled["ySpeed_body_mps"].values,
            self.flowSpeeds["xflowSpeed_body"].values
            * self.flowSpeeds["yflowSpeed_body"].values,
            mode="valid",
        )
        print(
            f"Length of flight log: {len(self.flightLogUpsampled)}, combined {len(self.flightLogUpsampled['xSpeed_body_mps'].values + self.flightLogUpsampled['ySpeed_body_mps'].values)}"
        )

        # Plot cross correlation
        plt.figure()
        plt.plot(xCorr, label="X Speed Cross Correlation")
        plt.plot(yCorr, label="Y Speed Cross Correlation")
        plt.plot(xyCorr, label="XY Speed Cross Correlation")
        plt.xlabel("Lag")
        plt.ylabel("Cross Correlation")
        plt.legend()
        plt.show()

        # Find the maximum correlation
        xMaxCorr = np.argmax(xCorr[: len(xCorr) // 2])
        # Find all peaks in xCorr
        peaks, _ = find_peaks(xCorr)
        print(f"Peaks in X Speed Cross Correlation: {peaks}")
        peaks, _ = find_peaks(yCorr)
        print(f"Peaks in Y Speed Cross Correlation: {peaks}")
        yMaxCorr = np.argmax(yCorr[: len(yCorr) // 2])
        xyMaxCorr = np.argmax(xyCorr[: len(xyCorr) // 2])
        print(f"X Speed Max Correlation: {xMaxCorr}")
        print(f"Y Speed Max Correlation: {yMaxCorr}")
        print(f"XY Speed Max Correlation: {xyMaxCorr}")

        if self.selectedCorr is None:
            selectedCorr = 0
            print("WARN: using default correlation lag of 0")
        else:
            selectedCorr = self.selectedCorr

        # Add lag to data by adding empty rows
        emptyRow = {
            "xflowSpeed_body": np.nan,
            "yflowSpeed_body": np.nan,
            "frameIdx": np.nan,
        }
        self.paddingBefor = pd.DataFrame([emptyRow] * selectedCorr)
        self.paddingAfter = pd.DataFrame(
            [emptyRow]
            * (len(self.flightLogUpsampled) - len(self.flowSpeeds) - selectedCorr)
        )
        self.flowSpeedsPadded = pd.concat(
            [self.paddingBefor, self.flowSpeeds, self.paddingAfter], ignore_index=True
        )

        # Merge the data
        self.videoMetadata = self.flightLogUpsampled
        self.videoMetadata["xflowSpeed_body"] = self.flowSpeedsPadded[
            "xflowSpeed_body"
        ].values
        self.videoMetadata["yflowSpeed_body"] = self.flowSpeedsPadded[
            "yflowSpeed_body"
        ].values
        self.videoMetadata["frameIdx"] = self.flowSpeedsPadded["frameIdx"].values

        # Fill in missing frame indices
        # Start from first non-NaN value and fill backwards until zero
        firstNonNan = self.videoMetadata["frameIdx"].first_valid_index()
        firstNonNan = self.videoMetadata.index.get_loc(firstNonNan)
        if (
            self.start_frame > firstNonNan
        ):  # Handle the case where video starts before flight log
            self.videoMetadata.loc[
                self.videoMetadata.index[0:firstNonNan], "frameIdx"
            ] = range(self.start_frame - firstNonNan, self.start_frame)
        else:
            self.videoMetadata.loc[
                self.videoMetadata.index[firstNonNan - self.start_frame : firstNonNan],
                "frameIdx",
            ] = range(self.start_frame)

        # Start from last non-NaN value and fill forwards until end
        lastNonNan = self.videoMetadata["frameIdx"].last_valid_index()
        lastNonNan = self.videoMetadata.index.get_loc(lastNonNan)
        self.videoMetadata.loc[
            self.videoMetadata.index[
                lastNonNan : lastNonNan + (self.frameCount - self.end_frame)
            ],
            "frameIdx",
        ] = range(self.end_frame, self.frameCount)

        # Remove flight log data that is not in the video
        self.videoMetadata = self.videoMetadata.dropna(subset=["frameIdx"])

        self.plotSynchronizedSpeeds()

    def saveSynchronizedData(self):
        name = self.video_path.stem
        self.outputPathVideo = self.video_path.with_name(f"{name}_videoTelem.csv")
        self.outputPathPickle = self.video_path.with_name(f"{name}_videoTelem.pkl")

        # Save synchronized telemetry data
        # print(self.videoMetadata.head())
        self.videoMetadata.to_csv(
            self.outputPathVideo, date_format="%Y-%m-%d %H:%M:%S.%f%z"
        )

        # Pickle processing steps
        tmpCap = self.cap
        self.cap = None
        with open(self.outputPathPickle, "wb") as f:
            pickle.dump(self, f)
        self.cap = tmpCap

        print(
            f"Saved synchronized data to {self.outputPathVideo} and {self.outputPathPickle}"
        )

    def plotSynchronizedSpeeds(self):
        plt.figure()
        plt.plot(
            self.videoMetadata.index,
            self.videoMetadata["xflowSpeed_body"],
            label="Optical Flow X Speed",
        )
        plt.plot(
            self.videoMetadata.index,
            self.videoMetadata["yflowSpeed_body"],
            label="Optical Flow Y Speed",
        )
        plt.plot(
            self.videoMetadata.index,
            self.videoMetadata["xSpeed_body_mps"],
            label="FlightLog X",
        )
        plt.plot(
            self.videoMetadata.index,
            self.videoMetadata["ySpeed_body_mps"],
            label="FlightLog Y",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Speed [m/s]")
        plt.legend()
        plt.title("Synchronized Speeds")
        valid_indices = self.videoMetadata["xflowSpeed_body"].notna()
        plt.xlim(
            self.videoMetadata.index[valid_indices][0],
            self.videoMetadata.index[valid_indices][-1],
        )
        plt.savefig(self.video_path.with_name("syncFig.png"))
        plt.show()


if __name__ == "__main__":
    dataSet = Path("data/kenya/loc/raw")
    refPath = dataSet / "refDrone"
    csvPath = next(
        path
        for path in refPath.iterdir()
        if path.suffix == ".csv" and not path.name.endswith("_videoTelem.csv")
    )
    videoPath = next(path for path in refPath.iterdir() if path.suffix == ".MP4")
    syncParamPath = dataSet / "syncParam.json"
    with open(syncParamPath, "r") as f:
        data = f.read()
        syncParam = json.loads(data)

    tracker = OpticalFlowTracker(
        videoPath,
        syncParam["startTime"],
        syncParam["endTime"],
        selectedCorr=syncParam["selectedCorr"],
    )
    tracker.loadFlightLog(csvPath)
    tracker.run()
    tracker.computeCrossCorrelation()
    tracker.saveSynchronizedData()
