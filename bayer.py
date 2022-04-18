from __future__ import print_function, division
import numpy as np
import sys
np.set_printoptions(threshold=np.inf)

def bayerMask(img, phase="rggb"):
    """Create a boolean mask for R, G and B that represents the camera sensor's filter array."""
    if len(phase) != 4:
        raise "Must have 4 characters for phase"
    redPhaseT = np.empty(4)
    grnPhaseT = np.empty(4)
    bluPhaseT = np.empty(4)
    for i in range(len(phase)):
        if phase[i] == "r":
            redPhaseT[i] = True
            grnPhaseT[i] = False
            bluPhaseT[i] = False
        elif phase[i] == "g":
            redPhaseT[i] = False
            grnPhaseT[i] = True
            bluPhaseT[i] = False
        elif phase[i] == "b":
            redPhaseT[i] = False
            grnPhaseT[i] = False
            bluPhaseT[i] = True
    # Cope with colour or monochrome arrays
    c = 0
    if len(np.shape(img)) == 3:
        # colour
        h, w, c = np.shape(img)
    else:
        # monochrome
        h, w = np.shape(img)

    # Each colour is now a 1D array of 4 booleans.
    # Re-shape that into a 2x2 array and tile it in
    # X and Y so it's the same size as the original array.
    # The result is a True/False mask for the bayer position of each colour
    rp = np.tile(np.reshape(redPhaseT, (2, 2)), (h // 2, w // 2))
    gp = np.tile(np.reshape(grnPhaseT, (2, 2)), (h // 2, w // 2))
    bp = np.tile(np.reshape(bluPhaseT, (2, 2)), (h // 2, w // 2))

    # Convert integer 1's and 0's to boolean
    rp_bool = rp == 1
    gp_bool = gp == 1
    bp_bool = bp == 1

    return (rp_bool, gp_bool, bp_bool)


def bayer(rgb_img, phase="rggb"):
    """Create a monochrome array that represents what a camera sensor would capture."""
    rMask, gMask, bMask = bayerMask(rgb_img, phase)
    red = rgb_img[:, :, 0] * rMask
    grn = rgb_img[:, :, 1] * gMask
    blu = rgb_img[:, :, 2] * bMask
    bayer = red + grn + blu
    return bayer


def bayer_colour(rgb_img, phase="rggb"):
    """Create an array that represents what a camera sensor would capture, but leave it as an RGB image (for looking at)."""
    rMask, gMask, bMask = bayerMask(rgb_img, phase)
    red = rgb_img[:, :, 0] * rMask
    grn = rgb_img[:, :, 1] * gMask
    blu = rgb_img[:, :, 2] * bMask
    bayer = np.dstack((red, grn, blu))
    return bayer
