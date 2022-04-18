from __future__ import print_function, division
from PIL import Image
import numpy as np


def PIL2array(img):
    """Converts a PIL image into a numpy array"""
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)


def array2PIL(arr, size, mode="RGBA"):
    """Converts a numpy array into a PIL image"""
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255 * np.ones((len(arr), 1), np.uint8)]

    return Image.frombuffer(mode, size, arr.tostring(), "raw", mode, 0, 1)


def npShow(data, description="", scale=0):
    """Draw a numpy array as an image"""
    img = Image.fromarray(data)
    if scale != 0:
        if len(np.shape(img)) == 3:
            # colour
            h, w, c = np.shape(img)
        else:
            # black/white
            h, w = np.shape(img)
            c = 0
        img = img.resize((h * scale, w * scale), Image.NEAREST)
    img.show()
    if description != "":
        print(description)
    return

def npSave(data, description="", scale=0):
    """Draw a numpy array as an image"""
    img = Image.fromarray(data)
    path = "C:/Users/dawso.DESKTOP-SK6U4T8/Desktop/Thesis/pi_debayring_repo/" + description + ".png"
    if scale != 0:
        if len(np.shape(img)) == 3:
            # colour
            h, w, c = np.shape(img)
        else:
            # black/white
            h, w = np.shape(img)
            c = 0
        img = img.resize((h * scale, w * scale), Image.NEAREST)
    img.save(path)
    print("Saved Image: " + description)
    return
