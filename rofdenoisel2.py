# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 20:49:40 2021

@author: hanso
"""


import cv2
import numpy as np
from pylab import *
from PIL import Image

def gradient(u, u0, tau):
    du = cv2.Laplacian(u, cv2.CV_64F,ksize=3)
    return tau * (u - u0) - 2. * du


img = cv2.imread('Alloy_noisy.jpg', cv2.IMREAD_GRAYSCALE)
img = img.astype(np.uint8)

denoised_img = img.copy()
u0 = denoised_img
u  = denoised_img
tau = 0.125
maxiters = 2
for i in range(0, maxiters):
    step = tau*(exp(-i*2/maxiters)+0.1)
    g = gradient(u, u0, tau)
    gnorm = np.linalg.norm(g)
    u = u - step * gnorm
denoised_img=u

cv2.imshow('original image'  , img  )
cv2.imshow('denoised image'     , denoised_img    )

cv2.waitKey(0)
cv2.destroyAllWindows()