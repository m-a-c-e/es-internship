#!/usr/bin/python
# Author: Manan Patel, Carissa, Gbenga
# Date:   24 May 2022
# This script uses blob detection to find keypoints across
# a set of frames


# Standard imports
import cv2
from blob_detector_helpers import blob_detector
import numpy as np
import time

#############################    ~ VIDEO ~    MAIN CODE  #######################
start_time = time.time()

fps = 60
vid = cv2.VideoCapture('Stereovision_input_data/DatasetVid/vid2.MOV')

if(vid.isOpened() == False):
	print("Unable to open video stream")

# get frame resolution
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))

# define codec for the video we are writing
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MPEG'),fps,(frame_width, frame_height))

while(vid.isOpened()):
	ret,frame = vid.read()
	if ret == True:
		frame, res, _, _ = blob_detector(frame)
		out.write(frame)
	else :
		break

out.release()
vid.release()