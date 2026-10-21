import cv2  # noqa: I001
import itertools
from pathlib import Path

from dataProcessor import DataProcessor
from dataStructure import LabelledDroneImage

COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 0, 128),  # Purple
    (128, 0, 0),  # Maroon
    (128, 128, 0),  # Olive
    (0, 128, 0),  # Dark Green
    (0, 0, 128),  # Navy
    (255, 165, 0),  # Orange
    (255, 192, 203),  # Pink
    (210, 105, 30),  # Chocolate
    (75, 0, 130),  # Indigo
    (173, 255, 47),  # Green Yellow
    (192, 192, 192),  # Silver
    (255, 20, 147),  # Deep Pink
]


def displayMatchedImages(
    main: LabelledDroneImage,
    ref: LabelledDroneImage,
    winName="Matched Images",
    zoom_factor=1,
    savePath=None,
):
    # Zoom into the center of the image by cropping before resizing
    def crop_center(img, cropx, cropy):
        y, x = img.shape[0:2]
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty : starty + cropy, startx : startx + cropx]

    main_cropx = main.image.shape[1] // zoom_factor
    main_cropy = main.image.shape[0] // zoom_factor
    ref_cropx = ref.image.shape[1] // zoom_factor
    ref_cropy = ref.image.shape[0] // zoom_factor

    main_img_cropped = crop_center(main.image, main_cropx, main_cropy)
    ref_img_cropped = crop_center(ref.image, ref_cropx, ref_cropy)

    height = 3000
    # Use cropped images for display
    mainImg = cv2.resize(
        main_img_cropped,
        (int(main_img_cropped.shape[1] / main_img_cropped.shape[0] * height), height),
    )
    refImg = cv2.resize(
        ref_img_cropped,
        (int(ref_img_cropped.shape[1] / ref_img_cropped.shape[0] * height), height),
    )
    combinedImg = cv2.hconcat([mainImg, refImg])

    # Generate a list of distinct colors (BGR)

    color_cycle = itertools.cycle(COLORS)

    # Calculate cropping offsets
    main_offset_x = (main.image.shape[1] - main_cropx) // 2
    main_offset_y = (main.image.shape[0] - main_cropy) // 2
    ref_offset_x = (ref.image.shape[1] - ref_cropx) // 2
    ref_offset_y = (ref.image.shape[0] - ref_cropy) // 2

    for mainObj, refObj in zip(main.objects.values(), ref.objects.values()):
        color = next(color_cycle)

        # Adjust coordinates for cropping
        main_x_cropped = mainObj.imageX - main_offset_x
        main_y_cropped = mainObj.imageY - main_offset_y
        ref_x_cropped = refObj.imageX - ref_offset_x
        ref_y_cropped = refObj.imageY - ref_offset_y

        # Only draw if point is within cropped area
        if (
            0 <= main_x_cropped < main_cropx
            and 0 <= main_y_cropped < main_cropy
            and 0 <= ref_x_cropped < ref_cropx
            and 0 <= ref_y_cropped < ref_cropy
        ):
            # Calculate scaling factors for cropped images
            main_scale = mainImg.shape[1] / main_cropx
            ref_scale = refImg.shape[1] / ref_cropx

            # Draw dot on main image
            main_x_disp = int(main_x_cropped * main_scale)
            main_y_disp = int(main_y_cropped * main_scale)
            cv2.circle(combinedImg, (main_x_disp, main_y_disp), 20, color, -1)

            # Draw dot on ref image (shifted by main image width)
            ref_x_disp = int(ref_x_cropped * ref_scale) + mainImg.shape[1]
            ref_y_disp = int(ref_y_cropped * ref_scale)
            cv2.circle(combinedImg, (ref_x_disp, ref_y_disp), 20, color, -1)

            # Draw line between points
            cv2.line(
                combinedImg,
                (main_x_disp, main_y_disp),
                (ref_x_disp, ref_y_disp),
                color,
                7,
            )

    if savePath:
        cv2.imwrite(savePath, combinedImg)
    else:
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.imshow(winName, combinedImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def draw_object_points_and_lines(
    labelled_img: LabelledDroneImage, winName="Object Points", savePath=None
):
    img = labelled_img.image.copy()

    color_cycle = itertools.cycle(COLORS)
    obj_list = list(labelled_img.objects.values())

    # Connect even-indexed objects with the next odd-indexed one
    for i in range(0, len(obj_list) - 1, 2):
        color = next(color_cycle)
        obj1 = obj_list[i]
        obj2 = obj_list[i + 1]

        x1_disp = int(obj1.imageX)
        y1_disp = int(obj1.imageY)
        x2_disp = int(obj2.imageX)
        y2_disp = int(obj2.imageY)

        cv2.circle(img, (x1_disp, y1_disp), 20, color, -1)
        cv2.circle(img, (x2_disp, y2_disp), 20, color, -1)
        cv2.line(img, (x1_disp, y1_disp), (x2_disp, y2_disp), color, 7)

    if savePath:
        cv2.imwrite(savePath, img)
    else:
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.imshow(winName, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    dataFolder = Path("data/kenya/loc/")
    processor = DataProcessor(dataFolder)
    data = processor.loadLatestProcessedData("labelled_images")

    displayMatchedImages(
        data["mainLabelled"][2],
        data["refLabelled"][2],
        winName="Zebra Size - Main and Ref Images",
        savePath=Path("figures/fig09-KENYA_dataCollection/sampleMatched_tmp.png"),
    )
    
    dataFolder = Path("data/kenya/size")
    processor = DataProcessor(dataFolder)
    data = processor.loadLatestProcessedData("labelled_images")
    draw_object_points_and_lines(
        data["mainLabelled"][2],
        winName="Zebra Size - Main Image",
        savePath=Path("figures/fig09-KENYA_dataCollection/sampleSize.png"),
    )
