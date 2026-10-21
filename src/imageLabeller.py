import cv2 # noqa: I001
import tkinter as tk
from tkinter import simpledialog
from tqdm import tqdm

from dataStructure import DroneImage, LabelledDroneImage, Object, ColorPalette


class ImageLabeller:
    def __init__(self):
        self.colorPalette = ColorPalette()
        self.winWidth = 2000
        self.winHeight = 1500
        self.idCounter = 1

    def labelSingleImages(
        self, images: list[DroneImage], labelsList: list[str], showDialog: bool = True
    ):
        labelledImages = []
        winName = f"Image {images[0].param.fileName}"
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winName, self.winWidth, self.winHeight)

        for image in tqdm(images, desc="Labelling images"):
            labelledImage, key = self.doLabel(image, labelsList, winName, showDialog)
            labelledImages.append(labelledImage)
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        return labelledImages

    def doLabel(
        self,
        image: DroneImage,
        labelList: list[str],
        winName: str,
        showDialog: bool = True,
    ):
        doLabel = True
        while doLabel:
            self.idCounter = 0
            labelledImage = LabelledDroneImage(image, {})
            labelledImage.show(winName, showDescription=True)
            if len(labelList) == 0:
                print("No labels available for this image.")
                key = None
                break
            labels = iter(labelList)

            # print(f"Current label: {labelList[self.idCounter]}")
            def click_event(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if showDialog:
                        # Ask for ID and description
                        root = tk.Tk()
                        root.withdraw()  # Hide the main window
                        objId = simpledialog.askinteger(
                            "Input", "Enter ID:", initialvalue=self.idCounter
                        )
                        self.idCounter += 1
                        description = simpledialog.askstring(
                            "Input", "Enter description:", initialvalue=next(labels)
                        )
                        root.destroy()
                    else:
                        objId = self.idCounter
                        self.idCounter += 1
                        description = next(labels)

                    # Create object
                    color = self.colorPalette.getColor(objId, returnPairs=True)
                    obj = Object(
                        id=objId,
                        imageX=x,
                        imageY=y,
                        description=description,
                        color=color,
                    )
                    labelledImage.objects[objId] = obj

                    # Update the image
                    labelledImage.show(winName, showDescription=True)

                    # print(f"Next label: {labelList[self.idCounter%len(labelList)]}")

            cv2.setMouseCallback(winName, click_event)

            key = cv2.waitKey(0)
            if key == ord("q") or key == ord("s"):  # Quit OR save
                doLabel = False
            else:
                pass

        return labelledImage, key

    def labelSideBySide(
        self,
        images: list[DroneImage],
        refImages: list[LabelledDroneImage],
        labelList: list[str] | None = None,
    ):
        winImages = "Images to Label"
        cv2.namedWindow(winImages, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winImages, self.winWidth, self.winHeight)

        winRefImages = "Already Labelled Images"
        cv2.namedWindow(winRefImages, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winRefImages, self.winWidth, self.winHeight)

        labelledImages = []
        noneLabelList = labelList is None

        for image, refImage in zip(images, refImages):
            print(f"Showing image {image.param.fileName}")
            refImage.show(winRefImages)
            if noneLabelList:
                labelList = [obj.description for obj in refImage.objects.values()]

            labelledImage, key = self.doLabel(
                image, labelList, winImages, showDialog=False
            )
            if key == ord("q"):
                break
            labelledImages.append(labelledImage)

        cv2.destroyAllWindows()
        return labelledImages

    def copyLabels(
        self, sourceImages: list[LabelledDroneImage], targetImages: list[DroneImage]
    ):
        labelledImages = []
        for sourceImage, targetImage in tqdm(
            zip(sourceImages, targetImages),
            desc="Copying labels",
            total=len(sourceImages),
        ):
            # Calculate scale and offset
            # RGB and T image do not have the same aspect ratio and T image are cropped at the left side of the image
            # and offset when zoomed (don't ask...)
            # scaleYCorrection and offsetX scaling have been determined by trial and error
            if sourceImage.param.focalLengthIn35mmFormat == 24:  # Wide angle lense
                scaleY = targetImage.param.imageHeight / sourceImage.param.imageHeight
                scaleYCorrection = 1
                scaleX = scaleY
                offsetX = (sourceImage.param.digitalZoomRatio - 1) * -7
            elif sourceImage.param.focalLengthIn35mmFormat == 161:  # Tele lense
                scaleY = targetImage.param.imageHeight / sourceImage.param.imageHeight
                scaleYCorrection = 1.02
                scaleX = scaleY
                offsetX = (sourceImage.param.digitalZoomRatio - 1) * 27
            else:
                raise ValueError(
                    f"Unknown lense with 35mm focal length: {sourceImage.param.focalLengthIn35mmFormat}"
                )

            labelledImage = LabelledDroneImage(targetImage, {})
            for obj in sourceImage.objects.values():
                newObj = Object(
                    id=obj.id,
                    imageX=int(obj.imageX * scaleX + offsetX),
                    imageY=int(obj.imageY * scaleY * scaleYCorrection),
                    description=obj.description,
                    color=obj.color,
                )
                labelledImage.objects[newObj.id] = newObj
            labelledImages.append(labelledImage)
        return labelledImages

    def removeLabel(
        self, labelledImages: list[LabelledDroneImage], frameId: int, labelId: int
    ):
        lastObjectId = max(labelledImages[frameId].objects.keys())
        labelledImages[frameId].objects[labelId].imageX = (
            labelledImages[frameId].objects[lastObjectId].imageX
        )
        labelledImages[frameId].objects[labelId].imageY = (
            labelledImages[frameId].objects[lastObjectId].imageY
        )
        labelledImages[frameId].objects.pop(lastObjectId)
        return labelledImages

    def displayLabelledImages(self, images: list[LabelledDroneImage]):
        winName = f"Labelled images {images[0].param.fileName}"
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winName, self.winWidth, self.winHeight)

        for image in tqdm(images, desc="Displaying labelled images"):
            image.show(winName)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break

        cv2.destroyAllWindows()

    def displaySideBySide(
        self, mainImages: list[LabelledDroneImage], refImages: list[LabelledDroneImage]
    ):
        winImages = "Main Images"
        cv2.namedWindow(winImages, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winImages, self.winWidth, self.winHeight)

        winRefImages = "Ref Images"
        cv2.namedWindow(winRefImages, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winRefImages, self.winWidth, self.winHeight)
        for mainImage, refImage in tqdm(zip(mainImages, refImages)):
            mainImage.show(winImages, True, False)
            refImage.show(winRefImages, True, False)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
