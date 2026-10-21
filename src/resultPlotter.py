import cv2 # noqa: I001
import matplotlib.pyplot as plt
import rioxarray
import os
from PIL import Image
from pathlib import Path

from dataStructure import LocalisedDroneImage, geoLoc, ColorPalette, DroneImage

class ResultPlotter:
    def __init__(self, groundMapPath : Path):
        self.groundMapPath = groundMapPath
        self.loadGroundMap()
        self.colorPalette = ColorPalette()
        self.winWidth = 2000
        self.winHeight = 1500


    def loadGroundMap(self):
        self.groundMap = rioxarray.open_rasterio(self.groundMapPath)
        self.groundMapCRS = self.groundMap.rio.crs

    def displayLocalizedDroneImagesOnMap(self,
            mainImages : list[LocalisedDroneImage],
            refImages : list[LocalisedDroneImage] = [],
            showDroneLocation : bool = False,
            selectedObject : list[int] = [],
            showMain : bool = True,
            showRef : bool = True,
            colorIsID : bool = False,
            savePath : str | None = None,
            title : str = "",
            singleRef : geoLoc | None = None,
            residuals : list[float] | None = None,):
        plt.figure(figsize=(10, 10))
        self.groundMap.plot.imshow()

        if refImages == []:
            refImages = mainImages
            showRef = False

        imgIndex = 0
        for mainImage, refImage in zip(mainImages, refImages):
            for id, obj in mainImage.objects.items():
                if id not in selectedObject and len(selectedObject) > 0:
                    continue

                if showMain:
                    easting, northing = obj.loc.convertToCRS(self.groundMapCRS)
                    if colorIsID and residuals is None:
                        plt.plot(easting, northing, "o", color=self.colorPalette.getColor(id, maxIntensity=1), label=f"{obj.description} ({id})")
                    elif residuals is not None:
                        plt.scatter(easting, northing, c=[residuals[imgIndex]], cmap='plasma', vmin=min(residuals), vmax=max(residuals), label=f"{mainImage.param.fileName.split("_")[-1]}, obj:{id}")
                        imgIndex += 1
                    else:
                        plt.plot(easting, northing, "ro", label="Data")

                if showRef and id in refImage.objects:
                    refObj = refImage.objects[id]
                    refEasting, refNorthing = refObj.loc.convertToCRS(self.groundMapCRS)
                    plt.plot(refEasting, refNorthing, 'bo')
                    if showMain:
                        plt.plot([easting, refEasting], [northing, refNorthing], 'g-')

                if showDroneLocation:
                    mainDroneLoc = geoLoc(mainImage.param.GPSLatitude, mainImage.param.GPSLongitude)
                    mainEasting, mainNorthing = mainDroneLoc.convertToCRS(self.groundMapCRS)
                    plt.plot(mainEasting, mainNorthing, 'rx')
                    plt.plot([easting, mainEasting], [northing, mainNorthing], 'g--')

                if singleRef:
                    refEasting, refNorthing = singleRef.convertToCRS(self.groundMapCRS)
                    plt.plot(refEasting, refNorthing, 'bo', label="Ground Truth")


        plt.title(title)
        plt.xlabel(f"Easting (epsg:{self.groundMapCRS.to_epsg()})")
        plt.ylabel(f"Northing (epsg:{self.groundMapCRS.to_epsg()})")
        plt.axis('equal')
        # plt.legend()
        # TODO: fix lengend
        all_eastings = []
        all_northings = []

        for mainImage in mainImages:
            for id, obj in mainImage.objects.items():
                if id not in selectedObject and len(selectedObject) > 0:
                    continue
                easting, northing = obj.loc.convertToCRS(self.groundMapCRS)
                all_eastings.append(easting)
                all_northings.append(northing)

        for refImage in refImages:
            for id, obj in refImage.objects.items():
                if id not in selectedObject and len(selectedObject) > 0:
                    continue
                easting, northing = obj.loc.convertToCRS(self.groundMapCRS)
                all_eastings.append(easting)
                all_northings.append(northing)

        if all_eastings and all_northings:
            buffer = 10
            plt.xlim([min(all_eastings) - buffer, max(all_eastings) + buffer])
            plt.ylim([min(all_northings) - buffer, max(all_northings) + buffer])


        if savePath:
            plt.savefig(savePath, dpi=300)
            plt.close()
        else:
            plt.show()

    def displayLocalizedDroneImagesWithRef(self,
            mainImages : list[LocalisedDroneImage],
            refImages : list[LocalisedDroneImage] = [],
            showDroneLocation : bool = False,
            selectedObject : list[int] = [],
            showMain : bool = True,
            showRef : bool = True,
            colorIsID : bool = False,
            savePath : str | None = None,
            title : str = "",
            singleRef : geoLoc | None = None,):
        
        plt.figure(figsize=(30, 10))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(mainImages[0].image, cv2.COLOR_BGR2RGB))
        for id, obj in mainImages[0].objects.items():
            plt.plot(obj.imageX, obj.imageY, 'o', color=self.colorPalette.getColor(id, maxIntensity=1))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(refImages[0].image, cv2.COLOR_BGR2RGB))
        for id, obj in refImages[0].objects.items():
            plt.plot(obj.imageX, obj.imageY, 'o', color=self.colorPalette.getColor(id, maxIntensity=1))
        plt.axis('off')

        plt.subplot(1, 3, 3)

        if refImages == []:
            refImages = mainImages
            showRef = False

        for mainImage, refImage in zip(mainImages, refImages):
            for id, obj in mainImage.objects.items():
                if id not in selectedObject and len(selectedObject) > 0:
                    continue

                if showRef and id in refImage.objects:
                    refObj = refImage.objects[id]
                    refEasting, refNorthing = refObj.loc.convertToCRS(self.groundMapCRS)
                    plt.plot(refNorthing, refEasting, 'o', color=self.colorPalette.getColor(id, maxIntensity=1))

        # Compute xlim and ylim based on all objects
        all_eastings = []
        all_northings = []
        for mainImage in mainImages:
            for id, obj in mainImage.objects.items():
                if id not in selectedObject and len(selectedObject) > 0:
                    continue
                easting, northing = obj.loc.convertToCRS(self.groundMapCRS)
                all_eastings.append(easting)
                all_northings.append(northing)
        for refImage in refImages:
            for id, obj in refImage.objects.items():
                if id not in selectedObject and len(selectedObject) > 0:
                    continue
                easting, northing = obj.loc.convertToCRS(self.groundMapCRS)
                all_eastings.append(easting)
                all_northings.append(northing)
        if all_eastings and all_northings:
            xlim = [min(all_northings) - 10, max(all_northings) + 10]
            ylim = [min(all_eastings) - 10, max(all_eastings) + 10]
            plt.xlim(xlim)
            plt.ylim(ylim)
                   
        plt.xlabel(f"Northing (epsg:{self.groundMapCRS.to_epsg()})")
        plt.ylabel(f"Easting (epsg:{self.groundMapCRS.to_epsg()})")

        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.tight_layout()
   
        if savePath:
            plt.savefig(savePath, dpi=300)
            plt.close()
        else:
            plt.show()

    def createGifFromFolder(self, folderPath: str, outputGifPath: str, duration: int = 500):

        # Get all image files in the folder
        imageFiles = [os.path.join(folderPath, f) for f in os.listdir(folderPath) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        imageFiles.sort()  # Sort files to maintain order

        # Load images
        images = [Image.open(img) for img in imageFiles]

        # Save as GIF with reduced resolution
        low_res_images = [img.resize((img.width // 2, img.height // 2)) for img in images]
        if low_res_images:
            low_res_outputGifPath = os.path.splitext(outputGifPath)[0] + "_lowres.gif"
            low_res_images[0].save(low_res_outputGifPath, save_all=True, append_images=low_res_images[1:], duration=duration, loop=0)
        
        # Save as GIF
        if images:
            images[0].save(outputGifPath, save_all=True, append_images=images[1:], duration=duration, loop=0)
        else:
            print("No images found in the specified folder.")

    def displayGeoLocs(self, geoLocs : list[geoLoc], style : list[str] = ['ro']):
        plt.figure(figsize=(10, 10))
        self.groundMap.plot.imshow()

        for loc, s in zip(geoLocs, style):
            easting, northing = loc.convertToCRS(self.groundMapCRS)
            plt.plot(easting, northing, s)

        plt.title("")
        plt.xlabel(f"Easting (epsg:{self.groundMapCRS.to_epsg()})")
        plt.ylabel(f"Northing (epsg:{self.groundMapCRS.to_epsg()})")
        plt.axis('equal')
        plt.show()

    def displayImages(self, images : list[DroneImage]):
        winName = f'Image: {images[0].param.fileName}'
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winName, self.winWidth, self.winHeight)
        
        for image in images:
            image.show(winName)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()

    def plotErrorOverAllRuns(self, errorTable, savePath = None):
        # Extract data for plotting
        runs = range(len(errorTable["RUN"][:-1]))  # Exclude the "MEAN" row
        pitch_error = errorTable["Pitch Error [°]"][:-1]
        pitch_error_std = errorTable["Pitch Error Std [°]"][:-1]
        heading_error = errorTable["Heading Error [°]"][:-1]
        heading_error_std = errorTable["Heading Error Std [°]"][:-1]

        plt.figure()
        plt.errorbar(runs, pitch_error, yerr=pitch_error_std, fmt='-o', label="Pitch Error", capsize=5)
        plt.errorbar(runs, heading_error, yerr=heading_error_std, fmt='-o', label="Heading Error", capsize=5)

        plt.xlabel("Runs")
        plt.ylabel("Error [°]")
        plt.title("Pitch and Heading Errors with 1 Standard Deviations")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()


        if savePath:
            plt.savefig(savePath)
        else:
            plt.show()
