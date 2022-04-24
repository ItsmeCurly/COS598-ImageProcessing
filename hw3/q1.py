"""
Author: Adam Green <adam.green1@maine.edu>
Date: 21-03-2022


I could have used sitk.GetPixel() and sitk.SetPixel(), 
but I decided against it because I wanted to explore more 
into numpy's ndarrays since I have been needing to learn 
them for other things recently.

Also - it seems like sitk's GetImageFromArray implicitly 
converts every pixel value to Float64, so I could not figure
out a way to save the image besides going into ImageJ and 
saving them manually.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import SimpleITK as sitk

from scipy.signal import convolve2d


def _default_image():
    return sitk.Image(256, 256, sitk.sitkUInt8)


def normalize(arr):
    """
    Normalize the image between 0-255 pixel values.
    """
    # arr *= (255.0/arr.max())
    x, y = arr.shape
    for i in range(x):
        for j in range(y):
            if arr[j][i] < 0:
                arr[j][i] = 0
    return arr


def to_image(arr):
    """Simplifies the conversion to a sitk image from a numpy array."""
    arr = normalize(arr)
    return sitk.GetImageFromArray(arr)


def _isotropic_gaussian(
    a: int, b: int, σ: float, μ: tuple[float, float], x: float, y: float,
):
    """Computes the isotropic gaussian value at a certain point."""
    return (
        a
        * (1 / (2 * np.pi * σ ** 2))
        * np.exp(-0.5 * ((x - μ[0]) ** 2 + (y - μ[1]) ** 2) / (σ ** 2))
        + b
    )


def isotropic_gaussian(
    μ: tuple[float, float],
    σ: float,
    a: int = 1,
    b: Optional[np.ndarray] = sitk.GetArrayFromImage(_default_image()),
):
    """Computes the isotropic gaussian for an entire 2D-array (image)."""
    μ = np.asarray(μ)
    σ = np.asarray(σ)

    res = np.zeros(b.shape)
    for i, x in enumerate(b):
        for j, y in enumerate(x):
            # Set the pixel to the computed value
            res[i][j] = _isotropic_gaussian(a=a, b=y, σ=σ, μ=μ, x=i, y=j)
    return res


def _isotropic_gaussian_first(
    a: int, σ: float, μ: tuple[float, float], x: float, y: float,
):
    return (
        a
        * (1 / (2 * np.pi * σ ** 2))
        * ((x - μ[0]) / σ ** 2)
        * np.exp(-0.5 * ((x - μ[0]) ** 2 + (y - μ[1]) ** 2) / (σ ** 2))
    )


def isotropic_gaussian_first(
    μ: tuple[float, float],
    σ: float,
    a: int = 1,
    b: Optional[np.ndarray] = sitk.GetArrayFromImage(_default_image()),
):
    μ = np.asarray(μ)
    σ = np.asarray(σ)

    res = np.zeros(b.shape)
    for x, b_ in enumerate(b):
        for y, _ in enumerate(b_):
            # Set the pixel to the computed value
            res[y][x] = _isotropic_gaussian_first(a=a, σ=σ, μ=μ, x=x, y=y)
    return res


def convolution2d(image, kernel):
    """Applies a convolution with the given image and kernel."""
    m, n = kernel.shape
    if m == n:
        x, y = image.shape
        new_image = np.zeros((x, y))
        for i in range(x):
            for j in range(y):

                if i == 0 or i == x - m + 1 or j == 0 or j == y - m + 1:
                    new_image[j][i] = 0
                else:
                    new_image[j][i] = np.sum(image[j : j + m, i : i + m] * kernel)
    return new_image


image_viewer = sitk.ImageViewer()

# 1A: Image 1

gau_der_arr = isotropic_gaussian_first(μ=(128, 128), σ=20, a=100000)
img_1 = to_image(gau_der_arr)

# 1B1: Image 2
gau_arr = isotropic_gaussian(μ=(128, 128), σ=20, a=100000)
img_2 = to_image(gau_arr)


# 1B2: Image 3
arr2d = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=object) * 1 / 8
conv_arr = convolution2d(gau_arr, arr2d)
img_3 = to_image(conv_arr)

# 1C: Image 4
sub_arr = gau_der_arr - conv_arr
img_4 = to_image(sub_arr)


image_viewer.Execute(img_1)
image_viewer.Execute(img_2)
image_viewer.Execute(img_3)
image_viewer.Execute(img_4)
