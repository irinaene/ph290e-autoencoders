"""Preprocessing helper functions."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def center(x, y, w):
    centroid_x = np.average(x, weights=w)
    centroid_y = np.average(y, weights=w)
    
    x = x - centroid_x
    y = y - centroid_y
    
    return x, y

def rotate_princ_vertical(x, y, weights):
    if len(x) == 1:
        theta = np.arctan2(y[0], x[0])
    else:
        # compute PCA from covariance matrix
        x_y_mat = np.array([x, y])
        # weighted covariance
        wcov = np.cov(x_y_mat, aweights=weights)
        # eigenvalues and eigenvectors
        w, v = np.linalg.eig(wcov)
        # highest eigenval
        princ = v[:, np.argmax(w)]
        theta = np.arctan2(princ[1], princ[0])
    
    theta_rot = -theta + np.pi /2 
    x_rot = x * np.cos(theta_rot) - y * np.sin(theta_rot)
    y_rot = x * np.sin(theta_rot) + y * np.cos(theta_rot)
    
    return x_rot, y_rot

def flip(x, y, flip):
    if flip not in ["LR", "UD", "BOTH", "NONE"]:
        # panic mode
        print(f"Flip type not defined: {flip}")
        sys.exit(1)
    if flip == "LR":
        x *= -1
    elif flip == "UD":
        y *= -1
    elif flip == "BOTH":
        x *= -1
        y *= -1
    
    return x, y

def determine_flip(x, y, w):
    # find the index of the maximum intensity
    ind_max = np.argmax(w)
    x_max = x[ind_max]
    y_max = y[ind_max]
    
    flip_types = []
    if x_max < 0:
        flip_types.append("LR")
    if y_max < 0:
        flip_types.append("UD")
    
    if len(flip_types) == 0:
        return "NONE"
    elif len(flip_types) == 2:
        return "BOTH"
    return flip_types[0]

def normalize(x):
    total = np.sum(x)
    if total == 0:
        total = 1
    return x / total

def pixelize(x, y, vals, nPix=40):
    pix_bins = np.linspace(-1, 1, nPix + 1)
    vals_pix, _, _ = np.histogram2d(x, y, bins=pix_bins, weights=vals)
    
    return vals_pix
