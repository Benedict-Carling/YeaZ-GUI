import skimage.exposure
import numpy as np
import cv2

from unet.neural_network import prediction, threshold
from unet.segment import segment

DATAPATH = "example_data"
OUTPUTPATH = "output_data"
FILEPATH = "IMG_0818.PNG"
# FILEPATH = "2020_3_19_frame_100_cropped.tif"


def saveImage(img, name):
    cv2.imwrite(OUTPUTPATH + "/" + name, img)


# Load images
image = cv2.imread(DATAPATH + "/" + FILEPATH)
print(image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image_gray)

# Equalise the exposure as recommended
im = skimage.exposure.equalize_adapthist(image_gray)
# im = im_float*1.0
print(im)
saveImage(im, "input.png")

# Prediction yeast cells
predctionOutput = prediction(im=im, is_pc=False)
saveImage(predctionOutput, "prediction.png")

# get the threshold to use in segmentation, you can also add a second argument here if desired
segvalue = np.linspace(1, 10, 10)
print(segvalue)

thresholdOutput = threshold(predctionOutput)
saveImage(thresholdOutput, "threshold.png")

for value in segvalue:
    # Get the segmentation Mask
    finalOutput = segment(thresholdOutput, predctionOutput, value)
    saveImage(finalOutput, f"segment_{value}.png")
