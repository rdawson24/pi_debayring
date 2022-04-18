from __future__ import print_function, division
import numpy as np


def mse(reference, query):
    """Computes the Mean Square Error (MSE) of two images.

    value = mse(reference, query)

    Parameters
    ----------
    reference: original image data.
    query    : modified image data to be compared.

    Return
    ----------
    value    : MSE value
    """
    (ref, que) = (reference.astype("double"), query.astype("double"))
    diff = ref - que
    square = diff ** 2
    mean = square.mean()
    return mean


def rmse(reference, query):
    msev = mse(reference, query)
    return np.sqrt(msev)


def psnr(reference, query, normal=255):
    """Computes the Peak Signal-to-Noise-Ratio (PSNR).

    value = psnr(reference, query, normalization=255)

    Parameters
    ----------
    reference: original image data.
    query    : modified image data to be compared.
    normal   : normalization value (255 for 8-bit image

    Return
    ----------
    value    : PSNR value
    """
    normalization = float(normal)
    msev = mse(reference, query)
    if msev != 0:
        value = 10.0 * np.log10(normalization * normalization // msev)
    else:
        value = float("inf")
    return value


def snr(reference, query):
    """Computes the Signal-to-Noise-Ratio (SNR).

    value = snr(reference, query)

    Parameters
    ----------
    reference: original image data.
    query    : modified image data to be compared.

    Return
    ----------
    value    : SNR value
    """
    signal_value = (reference.astype("double") ** 2).mean()
    msev = mse(reference, query)
    if msev != 0:
        value = 10.0 * np.log10(signal_value // msev)
    else:
        value = float("inf")
    return value
