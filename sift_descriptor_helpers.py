import numpy as np

# pytorch
import torch
import cv2
from torch.nn import functional as F
from torchvision import transforms as T

SOBEL_X_KERNEL = torch.tensor(
	[[[
		[-1, 0, 1],
		[-2, 0, 2],
		[-1, 0, 1]
	]]], dtype=torch.float32)

SOBEL_Y_KERNEL = torch.tensor(
	[[[
		[-1, -2, -1],
		[0, 0, 0],
		[1, 2, 1]
	]]], dtype=torch.float32)

T_img_to_tensor = T.ToTensor()           # PIL image to tensor


def generate_features(img, coords, win_size):
	"""
	Description:
		gets a list of features centred around coords by calling SIFT descriptor
	Args:
		img:	  color image
		coords:   numpy array of pixel coordinates around which to generate features
		win_size: width and height of the patch
	Returns:
		feats:    numpy array of features
	"""
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	feats = []
	for i in range(len(coords)):
		feats.append(get_SIFT_descriptor(img, coords[i], win_size))
	feats = np.array(feats)
	return feats


def compute_image_gradients(image_bw):
	"""Use convolution with Sobel filters to compute the image gradient at each
	pixel.
	Args:
		image_bw: A numpy array of shape (M,N) containing the grayscale image
	Returns:
		Ix: Array of shape (M,N) representing partial derivatives of image
			w.r.t. x-direction
		Iy: Array of shape (M,N) representing partial derivative of image
			w.r.t. y-direction
	"""
	Ix = None
	Iy = None

	# convert image to tensor
	image_tensor = T_img_to_tensor(image_bw)
	Ix = np.squeeze(np.array(F.conv2d(image_tensor, SOBEL_X_KERNEL, None, 1, 1, 1, 1)))
	Iy = np.squeeze(np.array(F.conv2d(image_tensor, SOBEL_Y_KERNEL, None, 1, 1, 1, 1)))

	return Ix, Iy


def get_magnitudes_and_orientations(Ix: np.ndarray, Iy: np.ndarray):
	"""
	This function will return the magnitudes and orientations of the
	gradients at each pixel location.
	Args:
		Ix: array of shape (m,n), representing x gradients in the image
		Iy: array of shape (m,n), representing y gradients in the image
	Returns:
		magnitudes: A numpy array of shape (m,n), representing magnitudes of
			the gradients at each pixel location
		orientations: A numpy array of shape (m,n), representing angles of
			the gradients at each pixel location. angles should range from
			-PI to PI.
	"""
	magnitudes = np.sqrt(np.square(Ix) + np.square(Iy))
	orientations = np.arctan2(Iy, Ix)

	return magnitudes, orientations


def get_gradient_histogram_vec_from_patch(
	window_magnitudes: np.ndarray,
	window_orientations: np.ndarray
) -> np.ndarray:
	""" Given patch, form a vector of gradient histograms.
	Key properties to implement:
	(1) a feature_width/4x feature_width/4 grid of cells, each feature_width/4. It is simply the terminology
		used in the feature literature to describe the spatial bins where
		gradient distributions will be described. The grid will extend
		feature_width/2 - 1 to the left of the "center", and feature_width/2 to
		the right. The same applies to above and below, respectively. 
	(2) each cell should have a histogram of the local distribution of
		gradients in 16 orientations. Appending these histograms together will
		give you feature_width x feature_width x 16. The histograms should be
		added to the feature vector left to right then row by row (reading
		order).
	Do not normalize the histogram here to unit norm -- preserve the histogram
	values. A useful function to look at would be np.histogram.
	Args:
		window_magnitudes: array representing gradient magnitudes of the
			patch
		window_orientations: array representing gradient orientations of
			the patch
	Returns:
		wgh: representing weighted gradient histograms
	"""
	window_orientations = np.round(window_orientations, decimals=5)
	feature_width       = np.shape(window_magnitudes)[0]
	grid_width 			= feature_width // 4     # assuming feature_width is divisible by 4

	wgh = []
	i = 0
	while i < feature_width:
		j = 0
		while j < feature_width:
			# slice the patch
			mag_slice = window_magnitudes[i:i + grid_width, j:j + grid_width].flatten()           # left to right
			ori_slice = window_orientations[i:i + grid_width, j:j + grid_width].flatten()         # left to right
			hist = np.histogram(ori_slice, bins=16, range=(-1 * np.pi, np.pi), weights=mag_slice)
			hist = hist[0]
			wgh.append(hist)
			j += grid_width
		i += grid_width

	wgh = np.array(wgh)
	wgh = wgh.flatten()
	wgh = np.reshape(wgh, (np.size(wgh), 1))

	return wgh


def get_feat_vec(
	magnitudes,
	orientations,
) -> np.ndarray:
	"""
	Description:
		generates a SIFT feature descriptor by creating a histogram of gradients
	Args:
		magnitudes:   a (feature_width x feature_width) array of magnitude of gradients
		orientations: a (feature_width x feature_width) array of orientation of gradients
	Returns:
		SIFT feature descriptor array
	"""
	fv = None
	wgh = get_gradient_histogram_vec_from_patch(magnitudes, orientations)

	# normalisation and square root
	divisor = np.linalg.norm(wgh)
	if divisor == 0:
		fv = wgh
	else:
		fv = wgh / np.linalg.norm(wgh)
		fv = fv ** 0.5
	return fv


def get_SIFT_descriptor(img_bw, coord, feature_width):
	"""
	Description:
		Generates a sift feature descriptor by cropping a path of size
		feature_width x feature_width around the specified coordinate
	Args:
		img_bw:  grayscale image
		coord:   (x, y) pixel coordinate
	Returns:
		SIFT feature descriptor of shape
	"""
	x_coord = coord[0]
	y_coord = coord[1]

	# crop a feature_width x feature_width window around coord
	img_crop = img_bw[y_coord - feature_width // 2 + 1: y_coord + feature_width // 2 + 1, 
					  x_coord - feature_width // 2 + 1: x_coord + feature_width // 2 + 1]

	# Compute image gradients using given sobel filter
	Ix, Iy = compute_image_gradients(img_crop)		# vecotrised!

	# Get the magnitudes and orientation
	magnitudes, orientation = get_magnitudes_and_orientations(Ix, Iy)

	# get the feature vector
	vec = get_feat_vec(magnitudes, orientation)
	vec = np.reshape(vec, np.size(vec))
	vec = vec / (np.sum(vec) + 0.00001)	

	return vec


def match_features(
	features1: np.ndarray,
	features2: np.ndarray):
	"""
	Description:
		given a set of features, performs one to one matching without
		repeating matches
	Args:
		features1: 	a numpy array of features
		features2: 	a numpy array of features
	Returns:	
	"""
	threshold = 0.0425
	features1 = np.array(features1)
	features2 = np.array(features2)
	matches   = []
	dists     = compute_feature_distances(features1, features2)
	min_value = 0
	if dists.size == 0:
		return []
	else:
		while(True):
			min_value = np.min(dists)
			indices = np.where(dists == min_value)
			x = indices[0][0]
			y = indices[1][0]
			dists[x, :] = 10 * np.ones((1, dists.shape[1]))	# set to threshold so that match is not repeated
			dists[:, y] = 10 * np.ones((1, dists.shape[0]))	# set to threshold so that match is not repeated
			if min_value < threshold:
				matches.append([x, y])
			else:
				break
	return np.array(matches)


def compute_feature_distances(
	features1: np.ndarray,
	features2: np.ndarray
) -> np.ndarray:
	"""
	This function computes a list of distances from every feature in one array
	to every feature in another.
	Using Numpy broadcasting is required to keep memory requirements low.
	Note: Using a double for-loop is going to be too slow. One for-loop is the
	maximum possible. Vectorization is needed.
	See numpy broadcasting details here:
		https://cs231n.github.io/python-numpy-tutorial/#broadcasting
	Args:
		features1: A numpy array of shape (n1,feat_dim) representing one set of
			features, where feat_dim denotes the feature dimensionality
		features2: A numpy array of shape (n2,feat_dim) representing a second
			set of features (n1 not necessarily equal to n2)
	Returns:
		dists: A numpy array of shape (n1,n2) which holds the distances (in
			feature space) from each feature in features1 to each feature in
			features2
	"""
	dists = []
	if len(features1) == 0 or len(features2) == 0:
		return np.array([])
	for row in features1:
		diff = features2 - row
		diff = np.square(diff)
		diff = np.sum(diff, axis=1)
		diff = np.sqrt(diff)
		dists.append(diff)
	dists = np.array(dists)
	return dists


def es_draw_matches_single(img, pts_l, pts_r, x, z, r):
	"""
	Description:
		Takes in left and right image and the matched pixel co-ordinates
		and draws them as a mosaic along with depth (x,z or r) if provided
	Args:
		img_l: BGR left image
		img_r: BGR riht image
		pts_l: pixel coordinates in the left image
		pts_r: pixel coordinates in the right image
		x:	   horizontal distance in meters from the left camera
		z:	   perpendicular distance from the baseline of cameras
		r:	   raidal distance from the left camera
	Returns:
		vis:   a mosaic depicting the matched points and distances as
			   specified
	"""
	np.random.seed(0)
	h,w 	  = img.shape[:2]
	vis 	  = img
	font      = cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 0.7
	color     = (255, 0, 0)
	thickness = 2
	for i in range(len(pts_l)):
		color    = tuple(np.random.randint(0, 255, 3).tolist())
		start_pt = (int(pts_l[i][0]), int(pts_l[i][1]))
		end_pt   = (int(pts_r[i][0]) + w, int(pts_r[i][1]))
		vis      = cv2.circle(vis, start_pt, 5, color, -1)
		vis      = cv2.circle(vis, end_pt,   5, color, -1)
		if r is not None:
			vis = cv2.putText(vis, "({:.2f})".format(r[i]), 
							 start_pt, font, fontScale, (0, 0, 0), thickness + 10, cv2.LINE_AA)
			vis = cv2.putText(vis, "({:.2f})".format(r[i]), 
							 start_pt, font, fontScale, (0, 255//10 * r[i] // 1, 255 - (255 // 10) * r[i] // 1), thickness, cv2.LINE_AA)
		elif x is not None and z is not None:
			vis = cv2.putText(vis, "({:.2f}, {:.2f})".format(x[i], z[i]), 
							 start_pt, font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
		else:
			vis = cv2.line(vis, start_pt, end_pt, color, 2)
	return vis


def es_draw_matches(img_l, img_r, pts_l, pts_r, x, z, r):
	"""
	Description:
		Takes in left and right image and the matched pixel co-ordinates
		and draws them as a mosaic along with depth (x,z or r) if provided
	Args:
		img_l: BGR left image
		img_r: BGR riht image
		pts_l: pixel coordinates in the left image
		pts_r: pixel coordinates in the right image
		x:	   horizontal distance in meters from the left camera
		z:	   perpendicular distance from the baseline of cameras
		r:	   raidal distance from the left camera
	Returns:
		vis:   a mosaic depicting the matched points and distances as
			   specified
	"""
	np.random.seed(0)
	h,w 	  = img_l.shape[:2]
	vis 	  = np.concatenate((img_l, img_r), axis=1)
	font      = cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 0.7
	color     = (255, 0, 0)
	thickness = 2
	for i in range(len(pts_l)):
		color    = tuple(np.random.randint(0, 255, 3).tolist())
		start_pt = (int(pts_l[i][0]), int(pts_l[i][1]))
		end_pt   = (int(pts_r[i][0]) + w, int(pts_r[i][1]))
		vis      = cv2.circle(vis, start_pt, 5, color, -1)
		vis      = cv2.circle(vis, end_pt,   5, color, -1)
		if r is not None:
			vis = cv2.putText(vis, "({:.2f})".format(r[i]), 
							 start_pt, font, fontScale, (0, 0, 0), thickness + 10, cv2.LINE_AA)
			vis = cv2.putText(vis, "({:.2f})".format(r[i]), 
							 start_pt, font, fontScale, (0, 255//10 * r[i] // 1, 255 - (255 // 10) * r[i] // 1), thickness, cv2.LINE_AA)
		elif x is not None and z is not None:
			vis = cv2.putText(vis, "({:.2f}, {:.2f})".format(x[i], z[i]), 
							 start_pt, font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
		else:
			vis = cv2.line(vis, start_pt, end_pt, color, 2)

	return vis