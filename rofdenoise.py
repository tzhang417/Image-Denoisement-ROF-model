# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:32:07 2021

@author: hanso
"""


import numpy as np
from gradient import forward_differences, forward_differences_conj, prox_project

def ROF(f,x,y,clambda):
    a = np.linalg.norm((f-x).flatten())**2/2
    b = np.sum(np.sqrt(np.sum(y**2,axis=2)).flatten())
    return a+clambda*b


def denoisel1(image, clambda, tau, iters=100):

    y = forward_differences(image)
    x = image.copy()

    vallog = np.zeros(iters)

    for i in range(iters):
        gradg = forward_differences(forward_differences_conj(y) - image)
        y = prox_project(clambda, y-tau*gradg)
        x = image-forward_differences_conj(y)
        vallog[i] = ROF(image, x, forward_differences(x), clambda)

    return x

