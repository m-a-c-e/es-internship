#!/usr/bin/python
# Author: Carissa Bush
# Date:   15 July 2022
# This script takes in a folder of JPEG images
# and converts them to a AVI video


# Standard imports
import cv2
from blob_detector_helpers import blob_detector
import time
import os

#############################    ~ VIDEO ~    MAIN CODE  #######################
start_time = time.time()

dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/07_20_2022/orlaco/comp_zed_cam/'
# dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/07_20_2022/orlaco/comp_zed_cam_v2/'
# dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/07_21_2022/stream_data_orlaco_v2/'
# dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/07_22_2022/stream_data_orlaco_v3/'
# dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/cloudy_data/cloudy_day_orlaco/cloudy_day_orlaco_v1/'
# dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/cloudy_data/cloudy_day_orlaco/cloudy_day_orlaco_v2/'


img_name = 'results/mosaic'
img_number = 0

img = cv2.imread(dataset_path + img_name + str(img_number) + '.jpg')
frame_height = img.shape[0]
frame_width = img.shape[1]

fps = 30

# define codec for the video we are writing
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame_width, frame_height))
for i in range(len(os.listdir(dataset_path))-1):
    out.write(img)
    img_number += 1
    img = cv2.imread(dataset_path + img_name + str(img_number) + '.jpg')

out.release()