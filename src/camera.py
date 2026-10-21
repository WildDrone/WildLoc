import pickle  # noqa: I001
import cv2 as cv
import numpy as np
from pathlib import Path

from dataStructure import LabelledDroneImage


class Camera:
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.repError = None
        self.frameShape = None

    def getFocalLength(self):
        if self.mtx is None:
            raise ValueError("Intrinsic camera matrix not loaded")

        if self.mtx[0][0] != self.mtx[1][1]:
            raise ValueError("Focal length is not the same in x and y direction")

        return self.mtx[0][0]

    def getCx(self):
        if self.mtx is None:
            raise ValueError("Intrinsic camera matrix not loaded")

        return self.mtx[0][2]

    def getCy(self):
        if self.mtx is None:
            raise ValueError("Intrinsic camera matrix not loaded")

        return self.mtx[1][2]

    def undistortCoords(self, x, y):
        # Undistort the coordinates
        newCameraMtx, _ = cv.getOptimalNewCameraMatrix(
            self.mtx, self.dist, self.frameShape, 1, self.frameShape
        )
        undistortedCoords = cv.undistortPoints(
            np.array([[x, y]], dtype=np.float32),
            self.mtx,
            self.dist,
            None,
            newCameraMtx,
        )
        return int(undistortedCoords[0][0][0]), int(undistortedCoords[0][0][1])

    def loadCamParam(self, filePath):
        with open(filePath, "rb") as f:
            data = pickle.load(f)
            self.mtx = data["camera_matrix"]
            self.dist = data["dist_coeffs"]
            self.repError = data["reprojection_error"]
            self.frameShape = data["frame_shape"]
            self.frameShape = (int(self.frameShape[0]), int(self.frameShape[1]))

    def isSameShape(self, shape):
        if self.frameShape is None:
            raise ValueError("Frame shape not loaded")

        return int(self.frameShape[0]) == int(shape[0]) and int(
            self.frameShape[1]
        ) == int(shape[1])

    def print(self):
        print("########################")
        print(f"NAME: {self.__class__.__name__}")
        print(f"Camera matrix:\n{self.mtx}")
        print(f"Distortion coefficients:{self.dist}")
        print(f"Reprojection error: {self.repError}")
        print(f"Frame shape: {self.frameShape}")
        print("########################")


class CamM3Tz(Camera):
    def __init__(self):
        super().__init__()
        self.loadCamParam(Path("data/cameraCalibration/M3Tz.pkl"))

class CamM3Tw(Camera):
    def __init__(self):
        super().__init__()
        self.loadCamParam(Path("data/cameraCalibration/M3Tw.pkl"))

class CamMini3(Camera):
    def __init__(self):
        super().__init__()
        self.loadCamParam(Path("data/cameraCalibration/Mini3.pkl"))


class CameraAll:
    def __init__(self):
        self.cam = {}
        self.cam["M3Tz"] = CamM3Tz()
        self.cam["M3Tw"] = CamM3Tw()
        self.cam["Mini3"] = CamMini3()

    def findCam(self, labelledImage: LabelledDroneImage):
        if (
            labelledImage.param.droneType == "M3T"
            and labelledImage.param.focalLengthIn35mmFormat == 161
        ):
            cam = "M3Tz"
        elif (
            labelledImage.param.droneType == "M3T"
            and labelledImage.param.focalLengthIn35mmFormat == 24
        ):
            cam = "M3Tw"
        elif labelledImage.param.droneType == "Mini 3":
            cam = "Mini3"
        else:
            raise ValueError(
                f"Unknown camera type: {labelledImage.param.droneType} with focal length {labelledImage.param.focalLengthIn35mmFormat}"
            )
        return cam

    def getFocalLength(self, labelledImage: LabelledDroneImage):
        if not self.cam[self.findCam(labelledImage)].isSameShape(
            (labelledImage.param.imageWidth, labelledImage.param.imageHeight)
        ):
            raise ValueError(
                f"Camera shape does not match the image shape, expected {self.cam[self.findCam(labelledImage)].frameShape} but got {(labelledImage.param.imageWidth, labelledImage.param.imageHeight)}"
            )

        return self.cam[self.findCam(labelledImage)].getFocalLength()

    def getCx(self, labelledImage: LabelledDroneImage):
        if not self.cam[self.findCam(labelledImage)].isSameShape(
            (labelledImage.param.imageWidth, labelledImage.param.imageHeight)
        ):
            raise ValueError("Camera shape does not match the image shape")

        return self.cam[self.findCam(labelledImage)].getCx()

    def getCy(self, labelledImage: LabelledDroneImage):
        if not self.cam[self.findCam(labelledImage)].isSameShape(
            (labelledImage.param.imageWidth, labelledImage.param.imageHeight)
        ):
            raise ValueError("Camera shape does not match the image shape")

        return self.cam[self.findCam(labelledImage)].getCy()

    def undistortCoords(self, labelledImage: LabelledDroneImage, x, y):
        if not self.cam[self.findCam(labelledImage)].isSameShape(
            (labelledImage.param.imageWidth, labelledImage.param.imageHeight)
        ):
            raise ValueError("Camera shape does not match the image shape")

        return self.cam[self.findCam(labelledImage)].undistortCoords(x, y)


if __name__ == "__main__":
    camAll = CameraAll()
    for cam in camAll.cam.values():
        cam.print()
