import cv2
import numpy as np
import matplotlib.pyplot as plt


def hpf(img):
    img = cv2.GaussianBlur(img, (7,7), 0)
    fft = np.fft.fft2(img, axes=(0,1))

    fft_shift = np.fft.fftshift(fft)
    
    mag = np.abs(fft_shift)
    spec = np.log(mag) / 20

    radius = 16
    mask = np.zeros_like(img)
    center_x = mask.shape[1] // 2
    center_y = mask.shape[0] // 2
    # Invert mask
    mask = 255-mask
    
    cv2.circle(mask, (center_x,center_y), radius, (255,255,255), -1)[0]
    # Blur mask
    mask2 = cv2.GaussianBlur(mask, (19,19), 0)

    # Apply mask
    fft_shift_masked = np.multiply(fft_shift,mask) / 255
    fft_shift_masked2 = np.multiply(fft_shift,mask2) / 255


    # Shift origin back to top left
    back_ishift = np.fft.ifftshift(fft_shift)
    back_ishift_masked = np.fft.ifftshift(fft_shift_masked)
    back_ishift_masked2 = np.fft.ifftshift(fft_shift_masked2)

    # Inverse fft
    img_back = np.fft.ifft2(back_ishift, axes=(0,1))
    img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
    img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0,1))

    # combine complex real and imaginary components to form image again
    img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
    img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)
    img_filtered2 = np.abs(img_filtered2).clip(0,255).astype(np.uint8)
    
    return spec, img_filtered2

def lpf(img):
    img = cv2.GaussianBlur(img, (7,7), 0)
    fft = np.fft.fft2(img, axes=(0,1))

    fft_shift = np.fft.fftshift(fft)
    
    mag = np.abs(fft_shift)
    spec = np.log(mag) / 20

    radius = 16
    mask = np.zeros_like(img)
    center_x = mask.shape[1] // 2
    center_y = mask.shape[0] // 2
    cv2.circle(mask, (center_x,center_y), radius, (255,255,255), -1)[0]
    # Blur mask
    mask2 = cv2.GaussianBlur(mask, (19,19), 0)

    # Apply mask
    fft_shift_masked = np.multiply(fft_shift,mask) / 255
    fft_shift_masked2 = np.multiply(fft_shift,mask2) / 255


    # Shift origin back to top left
    back_ishift = np.fft.ifftshift(fft_shift)
    back_ishift_masked = np.fft.ifftshift(fft_shift_masked)
    back_ishift_masked2 = np.fft.ifftshift(fft_shift_masked2)

    # Inverse fft
    img_back = np.fft.ifft2(back_ishift, axes=(0,1))
    img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
    img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0,1))

    # combine complex real and imaginary components to form image again
    img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
    img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)
    img_filtered2 = np.abs(img_filtered2).clip(0,255).astype(np.uint8)
    
    return spec, img_filtered2

img = cv2.imread('einstein_NEW.png')
img2 = cv2.imread('marilyn.png')


spec1, filtered_img1 = lpf(img)
spec2, filtered_img2 = hpf(img2)

img_combined = cv2.addWeighted(filtered_img1, .5, filtered_img2, .5, 0)

cv2.imshow('img1', filtered_img1)
cv2.imshow('img2', filtered_img2)
cv2.imshow('img3', img_combined)
cv2.waitKey(0)

