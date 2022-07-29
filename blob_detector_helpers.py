#!/usr/bin/python

# Author: Manan Patel, Carissa, Gbenga
# Date:   24 May 2022
# This script uses blob detection to find keypoints which might
# correspond to orange/yellowish-green safety vest/hard hats colors

# Standard imports
import cv2
import numpy as np
import copy
from skimage.measure import regionprops,label


def get_bgr_ranges(color, delta):
	"""
	** Not used in the pipeline **
	Description:
		Based on color and delta, get the range of colors
	Args:
		color: 3x1 BGR numpy array
		delta: list of ranges for BGR ranges 
	Returns:
		high_color: color + delta
		low_color: color - delta
	"""
	high_color = np.array([min(color[0] + delta[0], 255), 
						   min(color[1] + delta[1], 255),
						   min(color[2] + delta[2], 255)])

	low_color  = np.array([max(color[0] - delta[0], 0), 
						   max(color[1] - delta[1], 0),
						   max(color[2] - delta[2], 0)])
	
	return (high_color.astype(np.int32), low_color.astype(np.int32))


def setup_blob_detector():
	"""
	** Not used in the pipeline **
	Description:
		Initializes the blob detector with necessary params
	Args:
		NA
	Returns:
		blob detector
	"""

	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	# Any value between the minThreshold and maxThreshold will
	# be selected
	params.minThreshold = 50 
	params.maxThreshold = 255

	# Fileter by Color
	params.filterByColor = True
	params.blobColor = 255 

	# Filter by Area.
	params.filterByArea = True
	params.minArea = 0.5 * 960
	params.maxArea = float('inf')

	# Filter by Circularity
	params.filterByCircularity = False

	# Filter by Convexity
	params.filterByConvexity = False
	params.minConvexity = 0.1
	params.maxConvexity = 1
		
	# Filter by Inertia
	params.filterByInertia = False
	# params.minInertiaRatio = 0.040
	# params.maxInertiaRatio = 1

	# Create a detector with the parameters
	ver = (cv2.__version__).split('.')
	detector = None
	if int(ver[0]) < 3 :
		detector = cv2.SimpleBlobDetector(params)
	else : 
		detector = cv2.SimpleBlobDetector_create(params)
	
	return detector


def inc_sat(hsv_img, value):
	"""
	Increases the saturation of the input image
	Args:
		img:   hsv image
		value: multiplier for saturation
	Returs:
		new_img: image with higher saturation
	"""
	hsv_img[:,:,1] = hsv_img[:, :, 1] * value
	hsv_img[:,:,1][hsv_img[:,:,1]>255] = 255
	hsv_img = np.array(hsv_img, dtype = np.uint8)

	return hsv_img


def color_mask(img, hue_low, hue_high, sat_low=100, sat_high=255,  value_low=150, value_high=255):
	"""
	Description:
		Takes in hsv image and lower limit and higher limit of hsv color value and returns
		a mask of the image
	Args:
		img: hsv image
		hue_low:	lower limit of hue
		hue_high:	higher limit of hue
		sat_low:	lower limit of saturation
		sat_high:	higher limit of saturation
		value_low:	lower limit of brightness
		value_high: higher limit of brightness
	Returns:
		mask: 		mask of the input image with the specified color range
	"""
	mask 	   = None
	image 	   = copy.copy(img)
	low_color  = np.array([hue_low, sat_low, value_low])
	high_color = np.array([hue_high, sat_high, value_high])
	mask  	   = cv2.inRange(image, low_color, high_color)
	return mask


def color_filter(img, hue_low, hue_high):
	mask = color_mask(img, hue_low=hue_low, hue_high=hue_high)
	res = cv2.bitwise_and(img, img, mask=mask)
	res    = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
	_, res = cv2.threshold(res, 10, 255, cv2.THRESH_BINARY)
	dilation_kernel = np.ones((15, 10), np.uint8)
	erosion_kernel  = np.ones((3, 3), np.uint8)
	res = cv2.erode(res, erosion_kernel, iterations=1)
	res = cv2.dilate(res, dilation_kernel, iterations=1)
	res = cv2.erode(res, erosion_kernel, iterations=1)
	res = cv2.dilate(res, dilation_kernel, iterations=2)

	return res


def blob_detector(res, minArea):
	# regionprops - gets the centroids, bounding boxes around blob
	label_image = label(res)
	regions     = regionprops(label_image)
	coords 	    = []
	boxes 		= []
	for props in regions:
		if props.area > minArea:
			y, x = props.centroid
			centre_coordinates = (int(x), int(y))
			coords.append(list(centre_coordinates))
			boxes.append([props.bbox[1], props.bbox[0], props.bbox[3], props.bbox[2]])
	coords = np.array(coords).astype(np.int32)

	return coords, boxes


def complete_blob(img, hue_low, hue_high):
	res = color_filter(img, hue_low, hue_high)
	coords, boxes = blob_detector(res, 0.5 * res.shape[0])
	return coords, boxes, res
 