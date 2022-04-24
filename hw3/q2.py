import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

def mask_image(img_name):
    img = cv2.imread('res\\39-53_2016.jpg', -1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, (0,0,0), (255, 255, 72)) 
    mask = cv2.bitwise_not(img, img, mask=mask1)
    return mask

img_gs = cv2.imread('res\\39-53_2016.jpg', 0)
img_bgr = cv2.imread('res\\39-53_2016.jpg', 1)

median = cv2.medianBlur(img_bgr,5)
median_gs = cv2.medianBlur(img_gs,5)

ret,th1 = cv2.threshold(median_gs,72,255,cv2.THRESH_BINARY)

th2 = cv2.adaptiveThreshold(median_gs,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

th3 = cv2.adaptiveThreshold(median_gs,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
gauss1 = cv2.GaussianBlur(img_bgr, (9,9), 0)
gauss2 = cv2.GaussianBlur(img_bgr, (5,5), 0)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
median = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
gauss1 = cv2.cvtColor(gauss1, cv2.COLOR_BGR2RGB)
gauss2 = cv2.cvtColor(gauss2, cv2.COLOR_BGR2RGB)

titles = [
    'Original Image - Grayscale',
    'Median Blur - Grayscale',
    'Global Thresholding (v = 127)',
    'Adaptive Mean Thresholding', 
    'Adaptive Gaussian Thresholding', 
    'Original Image',
    'Median Blur',
    "Gaussian Blur",
    "Gaussian Blur (smaller kernel)",
    "Image Mask",
]
images = [
    img_gs,
    median_gs,
    th1, 
    th2, 
    th3,
    img_rgb,
    median,
    gauss1,
    gauss2,
    cv2.cvtColor(mask_image("res\\39-53_2016.jpg"), cv2.COLOR_BGR2RGB)
]
nrows = ncols = math.ceil(math.sqrt(len(images)))

for i in range(len(images)):
    plt.subplot(nrows, ncols, i+1),plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
