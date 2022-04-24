import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

close_kernel = np.ones((7,7),np.uint8)

img1_bgr = cv2.imread('res\\AA9_sib_e41_6dpf_bf.png', cv2.IMREAD_COLOR)
img1_gs = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
img2_bgr = cv2.imread('res\\AA9_sib_e41_6dpf_bire.png', cv2.IMREAD_COLOR)
img2_gs = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img2_gs,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img2_gs_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=10)

contours, hierarchy = cv2.findContours(img2_gs_closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img = cv2.drawContours(img2_bgr, contours, -1, (0,255,0), 3)

cv2.imwrite("mut_e13.png", img)
titles = [
    'Original Image',
    'Masked Image',
    'Closed Image',
    'Contoured Image',
]
images = [
    img2_gs,
    thresh,
    img2_gs_closed,
    img,
]
nrows = ncols = math.ceil(math.sqrt(len(images)))

for i in range(len(images)):
    plt.subplot(nrows, ncols, i+1),plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
