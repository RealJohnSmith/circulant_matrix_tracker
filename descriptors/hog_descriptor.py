import cv2
import numpy as np
from numpy.linalg import norm
import math


class hog:
    cv_hog = None
    cellx = celly = 8
    bin_n = 16

def initialize(usegpu):
    hog.cv_hog = cv2.HOGDescriptor()

    return hog.bin_n


def describe(image):

    cellx = hog.cellx
    celly = hog.celly

    cellxCount = image.shape[1] / cellx
    cellyCount = image.shape[0] / celly
    cutOffX = image.shape[1] - cellxCount * cellx
    cutOffY = image.shape[0] - cellyCount * celly

    image = image[cutOffY / 2:-cutOffY / 2, cutOffX / 2:-cutOffX / 2, :]

    gx = cv2.Sobel(image[:, :, 0], cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(image[:, :, 0], cv2.CV_64F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin = np.int32(hog.bin_n * ang / (2 * np.pi) % hog.bin_n)

    bin_cells = []
    mag_cells = []

    for i in range(image.shape[0] / celly):
        for j in range(image.shape[1] / cellx):
            bin_cells.append(bin[i * celly:i * celly + celly - 1, j * cellx:j * cellx + cellx - 1])
            mag_cells.append(mag[i * celly:i * celly + celly - 1, j * cellx:j * cellx + cellx - 1])

    eps = 1e-7
    hists = [np.bincount(b.ravel(), m.ravel(), hog.bin_n) for b, m in zip(bin_cells, mag_cells)]

    desc = np.zeros([hog.bin_n, image.shape[0] / celly, image.shape[1] / cellx])

    for i in range(image.shape[0] / celly):
        for j in range(image.shape[1] / cellx):
            hist = hists[i * image.shape[1] / cellx + j]
            hist /= hist.sum() + eps
            hist = np.sqrt(hist)
            hist /= norm(hist) + eps
            desc[:, i, j] = hist

    return desc


def get_name():
    return "HOGFeatures"


def update_roi(old_roi, moved_by):
    roi = old_roi
    roi[0] = round(moved_by[1]) + roi[0]
    roi[1] = round(moved_by[0]) + roi[1]
    return roi
