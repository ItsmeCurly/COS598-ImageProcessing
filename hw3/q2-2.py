import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

img1_bgr = cv2.imread('res\\39-53_2017.jpg', -1)
img1_gs = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
img2_bgr = cv2.imread('res\\39-53_2016.jpg', -1)
img2_gs = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
img3_bgr = cv2.imread('res\\39-53_2014.jpg', -1)
img3_gs = cv2.cvtColor(img3_bgr, cv2.COLOR_BGR2GRAY)

img1_bgr = cv2.GaussianBlur(img1_bgr, (3,3), 0)
img2_bgr = cv2.GaussianBlur(img2_bgr, (3,3), 0)
img3_bgr = cv2.GaussianBlur(img3_bgr, (3,3), 0)

img1_mask = cv2.inRange(img1_gs, 100, 255)
img1_mask_inv = cv2.bitwise_not(img1_mask)

img1_no_shadows = cv2.bitwise_and(img1_bgr, img1_bgr, mask=img1_mask)
img2_repl_img1 = cv2.bitwise_and(img2_bgr, img2_bgr, mask=img1_mask_inv)
img1_result = cv2.add(img1_no_shadows, img2_repl_img1)

img2_mask = cv2.inRange(img2_gs, 100, 255)
img2_mask_inv = cv2.bitwise_not(img2_mask)

img2_no_shadows = cv2.bitwise_and(img2_bgr, img2_bgr, mask=img2_mask)
img1_repl_img2 = cv2.bitwise_and(img1_bgr, img1_bgr, mask=img2_mask_inv)
img2_result = cv2.add(img2_no_shadows, img1_repl_img2)

img_ref = cv2.addWeighted(img1_result, .5, img2_result, .5, 0)

titles = [
    'Image2 Overlayed on Image1 No Shadows',
    'Image1 Overlayed on Image2 No Shadows',
    'Image Averaged Result',
]
images = [
    cv2.cvtColor(img1_result, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(img2_result, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB),
]
nrows = ncols = math.ceil(math.sqrt(len(images)))

for i in range(len(images)):
    plt.subplot(nrows, ncols, i+1),plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

cv2.imwrite("out.png", img_ref)