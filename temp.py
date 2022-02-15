# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

import cv2
from PIL import Image
from pylab import *
from rofdenoise import denoisel1
from skimage import filters

img = cv2.imread('noisy2.png', cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32)
img /= 255.

clambda = 0.2
tau = 0.125
denoised_image   = denoisel1(img,clambda, tau, 500)

img = np.clip(img* 255., 0, 255).astype(np.uint8)

denoised_image   = np.clip(denoised_image   * 255., 0, 255).astype(np.uint8)

cv2.imshow('noisy image'  , img  )
cv2.imshow('denoised image'  , denoised_image  )
cv2.imshow("L2 denoised image", filters.gaussian(img, sigma=2))

cv2.waitKey(0)
cv2.destroyAllWindows()