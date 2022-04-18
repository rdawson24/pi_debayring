from __future__ import print_function, division
import sys
import numpy as np
from PIL import Image
import bayer
import debayerNN
import metrics
import bayerTools as bt
import matlab.engine

# reload(bayer)
# reload(debayerNN)

def main():
    roi = ()
    img = Image.open(sys.argv[1])
    roi = (80, 300, 100, 100)  # Region of Interest

    # data = PIL2array(img)
    data = np.array(img)[:, :, :3]
    data_crop = data[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]

    bt.npShow(data, "Original")
    #bt.npShow(data_crop, "Cropped Original", 8)

    if len(roi) == 5:
        ref = data_crop
    else:
        ref = data

    bayerPhase = "rggb"
    bayr = bayer.bayer(ref, bayerPhase)
    bayr_c = bayer.bayer_colour(ref, bayerPhase)

    bt.npShow(bayr_c, "Bayered (false colour)", 8)
    # b_img = bt.array2PIL(bayr, img.size)

    debay_img = debayerNN.debayer(bayr, bayerPhase).astype("uint8")
    bt.npShow(debay_img, "Nearest Neighbour Debayered", 8)
    bt.npSave(debay_img, "Output")

    diff = (ref.astype("int16") - debay_img.astype("int16")) + 127
    np.clip(diff, 0, 255, out=diff)
    diff_img = diff.astype("uint8")
    bt.npShow(diff_img, "Difference", 8)

    eng = matlab.engine.start_matlab()
    psnrResult = eng.PSNR(sys.argv[1])
    print("PSNR:", psnrResult)

    #print("PSNR R =", metrics.psnr(ref[0], diff_img[0]))
    #print("PSNR G =", metrics.psnr(ref[1], diff_img[1]))
    #print("PSNR B =", metrics.psnr(ref[2], diff_img[2]))
    #print("PSNR =", metrics.psnr(ref, diff_img))

if __name__ == '__main__':
    main()
