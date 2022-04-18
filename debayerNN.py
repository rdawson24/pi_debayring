from __future__ import print_function, division
import numpy as np
import bayer


def debayer(bayer_data, bayer_alignment="rggb"):
    """Implements a simple nearest-neighbour debayer

    Handles edges correctly by extending the border by 2 pixels on each
    side (by duplication), then trimming them again before returning
    """

    # This routine works using colour masked operations on shifted versions of the input array

    height, width = bayer_data.shape

    bayer_bordered = np.zeros((height + 4, width + 4), dtype=int)
    bayer_bordered[2:-2, 2:-2] = bayer_data  # copy the middle bit

    bayer_bordered[:2, 2:-2] = bayer_data[:2, :]  # duplicate the 2 top lines
    bayer_bordered[-2:, 2:-2] = bayer_data[-2:, :]  # duplicate the 2 bottom lines

    bayer_bordered[2:-2, :2] = bayer_data[:, :2]  # duplicate the 2 left lines
    bayer_bordered[2:-2, -2:] = bayer_data[:, -2:]  # duplicate the 2 bottom lines

    # bayer_bordered1 = np.pad(bayer_data, 2, 'symmetric')

    # bayer_bordered1[0, 0] = 0
    # bayer_bordered1[0, 1] = 0
    # bayer_bordered1[1, 0] = 0
    # bayer_bordered1[1, 1] = 0

    # bayer_bordered1[-1, 0] = 0
    # bayer_bordered1[-2, 0] = 0
    # bayer_bordered1[-1, 1] = 0
    # bayer_bordered1[-2, 1] = 0

    # bayer_bordered1[0, -1] = 0
    # bayer_bordered1[0, -2] = 0
    # bayer_bordered1[1, -1] = 0
    # bayer_bordered1[1, -2] = 0

    # bayer_bordered1[-1, -1] = 0
    # bayer_bordered1[-1, -2] = 0
    # bayer_bordered1[-2, -1] = 0
    # bayer_bordered1[-2, -2] = 0

    # bayer_bordered1[:, [0, 1]] = bayer_bordered1[:, [1, 0]]
    # bayer_bordered1[:, [-1, -2]] = bayer_bordered1[:, [-2, -1]]

    # bayer_bordered1[[0, 1], :] = bayer_bordered1[[1, 0], :]
    # bayer_bordered1[[-1, -2], :] = bayer_bordered1[[-2, -1], :]

    # print(bayer_bordered == bayer_bordered1)

    bayer_bordered_d = np.roll(bayer_bordered, 1, axis=0)
    bayer_bordered_u = np.roll(bayer_bordered, -1, axis=0)
    bayer_bordered_r = np.roll(bayer_bordered, 1, axis=1)
    bayer_bordered_l = np.roll(bayer_bordered, -1, axis=1)
    bayer_bordered_ur = np.roll(bayer_bordered_u, 1, axis=1)
    bayer_bordered_ul = np.roll(bayer_bordered_u, -1, axis=1)
    bayer_bordered_dr = np.roll(bayer_bordered_d, 1, axis=1)
    bayer_bordered_dl = np.roll(bayer_bordered_d, -1, axis=1)

    # -------------------------------------------------------------------------
    # pixel indices
    # -------------------------------------------------------------------------
    r_idx, g_idx, b_idx = bayer.bayerMask(
        bayer_bordered, bayer_alignment
    )  # red, green & blue pixel indices

    # Create a bordered Green-Red mask
    gr_idx = np.logical_and(
        g_idx,
        np.repeat(
            np.logical_or(r_idx[:, 0], r_idx[:, 1]), bayer_bordered.shape[1]
        ).reshape(bayer_bordered.shape),
    )  # green indices on green/red row
    gr_idx[:1, :] = False  # top line
    gr_idx[-1:, :] = False  # bottom line
    gr_idx[:, :1] = False  # left line
    gr_idx[:, -1:] = False  # right line

    # Create a bordered Green-Blue mask
    gb_idx = np.logical_and(
        g_idx,
        np.repeat(
            np.logical_or(b_idx[:, 0], b_idx[:, 1]), bayer_bordered.shape[1]
        ).reshape(bayer_bordered.shape),
    )  # green indices on green/blue row
    gb_idx[:1, :] = False  # top line
    gb_idx[-1:, :] = False  # bottom line
    gb_idx[:, :1] = False  # left line
    gb_idx[:, -1:] = False  # right line

    # Create a bordered not-Green mask
    ng_idx = np.logical_not(g_idx)  # non-green pixel indices
    ng_idx[:1, :] = False  # top line
    ng_idx[-1:, :] = False  # bottom line
    ng_idx[:, :1] = False  # left line
    ng_idx[:, -1:] = False  # right line

    # Border the original indexes
    r_idx[:1, :] = False  # top line
    r_idx[-1:, :] = False  # bottom line
    r_idx[:, :1] = False  # left line
    r_idx[:, -1:] = False  # right line
    g_idx[:1, :] = False  # top line
    g_idx[-1:, :] = False  # bottom line
    g_idx[:, :1] = False  # left line
    g_idx[:, -1:] = False  # right line
    b_idx[:1, :] = False  # top line
    b_idx[-1:, :] = False  # bottom line
    b_idx[:, :1] = False  # left line
    b_idx[:, -1:] = False  # right line

    red = np.zeros(bayer_bordered.shape, dtype=int)
    grn = np.zeros(bayer_bordered.shape, dtype=int)
    blu = np.zeros(bayer_bordered.shape, dtype=int)

    red[r_idx] = bayer_bordered[r_idx]
    red[gr_idx] = (
        bayer_bordered_l[gr_idx] + bayer_bordered_r[gr_idx]
    ) // 2  # Averaged either side
    red[gb_idx] = (bayer_bordered_u[gb_idx] + bayer_bordered_d[gb_idx]) // 2
    red[b_idx] = (
        bayer_bordered_ur[b_idx]
        + bayer_bordered_ul[b_idx]
        + bayer_bordered_dr[b_idx]
        + bayer_bordered_dl[b_idx]
    ) // 4  # Averaged diagonals

    grn[g_idx] = bayer_bordered[g_idx]
    #    grn[ng_idx] = (bayer_bordered_l[ng_idx] + bayer_bordered_r[ng_idx]) // 2 # Averaged either side
    grn[ng_idx] = (
        bayer_bordered_l[ng_idx]
        + bayer_bordered_r[ng_idx]
        + bayer_bordered_u[ng_idx]
        + bayer_bordered_d[ng_idx]
    ) // 4  # Averaged U,D,L,R

    blu[b_idx] = bayer_bordered[b_idx]
    blu[gb_idx] = (
        bayer_bordered_l[gb_idx] + bayer_bordered_r[gb_idx]
    ) // 2  # Averaged either side
    blu[gr_idx] = (bayer_bordered_u[gr_idx] + bayer_bordered_d[gr_idx]) // 2
    blu[r_idx] = (
        bayer_bordered_ur[r_idx]
        + bayer_bordered_ul[r_idx]
        + bayer_bordered_dr[r_idx]
        + bayer_bordered_dl[r_idx]
    ) // 4  # Averaged diagonals

    # combine rgb
    debayer = np.dstack((red, grn, blu))
    debayer_data = debayer.astype("uint8")  # if bpp <= 8 else 'uint16')
    #print(debayer_data[2:-2, 2:-2])
    # Return the center-crop
    return debayer_data[2:-2, 2:-2]
