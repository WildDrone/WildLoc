# More information here: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

import numpy as np  # noqa: I001
import cv2 as cv
import pickle
import tqdm
from pathlib import Path

# --- Global parameters ---
SHOW_FRAMES = True

# Chessboard size and square size in meters
GRID_SHAPE = (9, 14)
SQUARE_SIZE = 16.5e-3


def doCalibration(CALLIB_VIDEO, CAM_TYPE):
    """
    Perform camera calibration using a video file containing a chessboard pattern.
    :param CALLIB_VIDEO: Path to the video file.
    :param CAM_TYPE: Camera type (e.g., "M3Tz").
    """
    # --- Video file used for calibration ---

    # --- Preparing the 3D points of the chessboard ---
    # Generate a 3D points array for each corner of the grid
    objp = np.zeros((GRID_SHAPE[0] * GRID_SHAPE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : GRID_SHAPE[0], 0 : GRID_SHAPE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Storing points in two lists:
    #  - objpoints: 3D real-world points
    #  - imgpoints: 2D image points
    objpoints = []
    imgpoints = []

    # --- Window configuration for display ---
    if SHOW_FRAMES:
        cv.namedWindow("img", cv.WND_PROP_FULLSCREEN)
        cv.resizeWindow("img", 1280, 720)

    # --- Opening the video file ---
    cap = cv.VideoCapture(CALLIB_VIDEO)
    nbrDroppedFrames = 0

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    with tqdm.tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            # Read one frame from the video
            ret, frame = cap.read()

            if not ret:
                # End of the video or reading error
                break

            # Sharpen the image
            # kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
            # frame = cv.filter2D(frame, -1, kernel)
            # Blur the image
            s = 11
            frame = cv.GaussianBlur(frame, (s, s), 0)

            # Downsample the frame
            # r = 2
            # frame = cv.re size(frame, (0, 0), fx=1/r, fy=1/r)
            # # Upsample the frame
            # frame = cv.resize(frame, (0, 0), fx=r, fy=r)

            # Convert to grayscale
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Detect the corners of the chessboard
            # found_corners, corners = cv.findChessboardCornersSB(gray, GRID_SHAPE, flags=0)
            # found_corners, corners = cv.findChessboardCornersSB(gray, GRID_SHAPE, flags=cv.CALIB_CB_EXHAUSTIVE+cv.CALIB_CB_NORMALIZE_IMAGE)
            found_corners, corners = cv.findChessboardCornersSB(
                gray,
                GRID_SHAPE,
                flags=cv.CALIB_CB_EXHAUSTIVE
                + cv.CALIB_CB_ACCURACY
                + cv.CALIB_CB_NORMALIZE_IMAGE,
            )

            if found_corners:
                # Automatically store the frame if corners are found
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw the detected corners for visualization
                s = 0.2
                frameLowRes = cv.resize(frame, (0, 0), fx=s, fy=s)
                cv.drawChessboardCorners(frame, GRID_SHAPE, corners, found_corners)

                # Save low res frame for paper
                cv.drawChessboardCorners(
                    frameLowRes, GRID_SHAPE, corners * s, found_corners
                )
                savePath = Path(
                    f"figures/fig04-cameraCallib/calibPics/{CAM_TYPE}/{len(imgpoints):03d}.png"
                )
                savePath.parent.mkdir(parents=True, exist_ok=True)
                cv.imwrite(str(savePath), frameLowRes)
                pbar.set_postfix(
                    {
                        "Status": "Frame ADDED!",
                        "Success": len(imgpoints),
                        "Dropped": nbrDroppedFrames,
                        "Ratio": f"{len(imgpoints) / (len(imgpoints) + nbrDroppedFrames) * 100:.2f} %",
                    }
                )
            else:
                # If corners are not found, increment the dropped frames counter
                nbrDroppedFrames += 1
                pbar.set_postfix(
                    {
                        "Status": "Frame DROPPED!",
                        "Success": len(imgpoints),
                        "Dropped": nbrDroppedFrames,
                        "Ratio": f"{len(imgpoints) / (len(imgpoints) + nbrDroppedFrames) * 100:.2f} %",
                    }
                )

            # Update progress bar
            pbar.update(1)

            # Optional display
            if SHOW_FRAMES:
                cv.imshow("img", frame)

            # Keyboard interaction
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                # Quit when 'q' is pressed
                break

    streamShape = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    cv.destroyAllWindows()

    # --- Final calibration if at least one valid frame is available ---
    if len(objpoints) > 0 and len(imgpoints) > 0:
        # Compute camera parameters
        ret_calib, mtx, dist, rvecs, tvecs, stdIntrinsics, _, _ = (
            cv.calibrateCameraExtended(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                None,
                None,
                # flags=cv.CALIB_FIX_ASPECT_RATIO
                flags=cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_FIX_PRINCIPAL_POINT,
            )
        )

        # Display results
        print(f"RMS reprojection error: {ret_calib}")
        print("Camera matrix:")
        print(mtx)
        print("Stds of intrinsic parameters:")
        print(stdIntrinsics[:4])
        print("Distortion coefficients:")
        print(dist)

        # Save the calibration results using pickle
        calib_data = {
            "camera_matrix": mtx,
            "dist_coeffs": dist,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "reprojection_error": ret_calib,
            "frame_shape": streamShape,
        }
        savePath = Path(f"data/cameraCalibration/{CAM_TYPE}.pkl")
        savePath.parent.mkdir(parents=True, exist_ok=True)
        with open(savePath, "wb") as f:
            pickle.dump(calib_data, f)
    else:
        print("No valid frames were captured for calibration.")


if __name__ == "__main__":
    # Video file used for calibration
    CALLIB_VIDEO = {
        # "M3Tz": Path("data/cameraCalibration/calibrationPictures/M3Tz/%03d.JPG"),
        "M3Tw": Path("data/cameraCalibration/calibrationPictures/M3Tw/%03d.JPG"),
        # "Mini3": Path(
        #     "data/cameraCalibration/calibrationPictures/Mini3/ALL_SUBSAMPLE.MP4"
        # ),
    }

    # SUBSAMPLE CMD:
    # ffmpeg -i FRED.MP4 -vf "select=not(mod(n\,25))" -vsync vfr -c:v libx264 FRED_SUBSAMPLE.MP4

    for CAM_TYPE, video_path in CALLIB_VIDEO.items():
        doCalibration(video_path, CAM_TYPE)
