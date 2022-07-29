#!/usr/bin/python

# Author: Manan Patel
# Date:   24 May 2022
# This script uses blob detection to find keypoints in a frame


import cv2
from blob_detector_helpers import blob_detector
import numpy as np


frame_path = '/Users/manan.patelequipmentshare.com/Desktop/07_21_2022/stream_data_orlaco_v2/left-0.jpg'
img = cv2.imread(frame_path)
frame, mosaic, coords = blob_detector(img)
mosaic = np.concatenate((frame, mosaic), axis=1)
cv2.imwrite("mosaic.jpg", mosaic)