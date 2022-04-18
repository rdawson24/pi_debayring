# -*- coding: utf-8 -*-
"""
Extend the standard Numpy array manipulation routines

Copyright Iain Waugh, iwaugh@gmail.com
"""

from __future__ import print_function, division
import numpy as np


def shift_2d(arr, shifts, stride=0):
    """
    Shift a 2D array by rows/columns and either pad the new data with zeros or
    copy existing data from the trailing edge of the shift.

    Parameters:
    arr - 2D numpy array

    shift_row - how many rows to shift up/down

    shift_col - how many columns to shift left/right

    stride - how many of the previous rows/columns to repeat

    0 = fill with Zeros

    1 = copy the last row/column

    2 = copy the last row/column (good for working with Bayer paterns)

    Returns:
    An array that is shifted by 'shift_row' and 'shift_col', with new data determined by 'stride'
    """
    (shift_row, shift_col) = shifts
    row_arr = np.zeros_like(arr)
    if stride > 0:
        quot = abs(shift_row) // stride
        rem = shift_row % stride
        if rem != 0:
            quot += 1

    if shift_row > 0:
        if stride > 0:
            stride_arr = np.tile(arr[:stride, :], (quot, 1))[-shift_row:, :]
            row_arr = np.vstack((stride_arr, arr[:-shift_row, :]))
        else:
            row_arr[shift_row:, :] = arr[:-shift_row, :]

    elif shift_row < 0:
        if stride > 0:
            stride_arr = np.tile(arr[-stride:, :], (quot, 1))[:-shift_row, :]
            row_arr = np.vstack((arr[-shift_row:, :], stride_arr))
        else:
            row_arr[:shift_row, :] = arr[-shift_row:, :]
    else:
        row_arr[:, :] = arr[:, :]

    # Shift the Columns
    if stride > 0:
        quot = abs(shift_col) // stride
        rem = shift_col % stride
        if rem != 0:
            quot += 1
    new_arr = np.zeros_like(arr)
    if shift_col > 0:
        if stride > 0:
            stride_arr = np.tile(row_arr[:, :stride], (1, quot))[:, -shift_col:]
            new_arr = np.hstack((stride_arr, row_arr[:, :-shift_col]))
        else:
            new_arr[:, shift_col:] = row_arr[:, :-shift_col]

    elif shift_col < 0:
        if stride > 0:
            stride_arr = np.tile(row_arr[:, -stride:], (1, quot))[:, :-shift_col]
            new_arr = np.hstack((row_arr[:, -shift_col:], stride_arr))
        else:
            new_arr[:, :shift_col] = row_arr[:, -shift_col:]
    else:
        new_arr[:, :] = row_arr[:, :]

    return new_arr


if __name__ == "__main__":
    aRow, aCol = (6, 6)  #  Number of rows and columns
    arr = np.arange(1, aRow * aCol + 1).astype("uint16").reshape(aRow, aCol)

    n1 = shift_2d(arr, (1, 0), 1)
    s1 = shift_2d(arr, (-1, 0), 1)
    w1 = shift_2d(arr, (0, 1), 1)
    e1 = shift_2d(arr, (0, -1), 1)

    print("North 1\n", n1)
    print("South 1\n", s1)
    print("East 1\n", e1)
    print("West 1\n", w1)

    n2 = shift_2d(arr, (2, 0), 2)
    s2 = shift_2d(arr, (-2, 0), 2)
    w2 = shift_2d(arr, (0, 2), 2)
    e2 = shift_2d(arr, (0, -2), 2)

    print("North 2\n", n2)
    print("South 2\n", s2)
    print("East 2\n", e2)
    print("West 2\n", w2)
