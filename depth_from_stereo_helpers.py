#!/usr/bin/python

# Author: Manan Patel, Gbenga Omotara, Carissa Bush
# Date:   6 June 2022
# Description:
# This script contains functions for calculation depth from 
# a pair of keypoints


import numpy as np
import cv2
import glob
from   sift_descriptor_helpers import *
from   scipy import linalg
import scipy.io


def es_generate_calib_files(images_folder):
    """
    ** Not used in the pipeline **
    Description:
    Args:
    Returns:
    """
    ret   = None
    mtx   = None
    dist  = None 
    rvecs = None 
    tvecs = None
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv2.imread(imname, 1)
        images.append(im)

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = 7 #number of checkerboard rows. (8)
    columns = 10 #number of checkerboard columns.(11)
    world_scaling = 1. #change this to the real world square size. Or not.

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    iter = 0

    for frame in images:
        iter+=1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (rows,columns), corners, ret)
            image_name = "img_" + str(iter) + ".jpg"
            cv2.imwrite(image_name,frame)
            #cv.imshow('img', frame)
            #k = cv.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    
    np.save('camera_matrix.npy',mtx)
    np.save('distortion_coefficients.npy',dist)


def read_calib_files(folder_path):
    """
    Description:
        Reads and returns the calibration matrix from the
        specified file_path
    Args:
        file_path: Path to .mat file
    Returns:
        K:  camera intrinsics (numpy array)
        D: distortion coefficients (numpy array)
    """

    K1 = scipy.io.loadmat(folder_path+'K1.mat')
    K2 = scipy.io.loadmat(folder_path+'K2.mat')
    D1 = scipy.io.loadmat(folder_path+'D1.mat')
    D2 = scipy.io.loadmat(folder_path+'D2.mat')
    R = scipy.io.loadmat(folder_path+'R.mat')
    t = scipy.io.loadmat(folder_path+'T.mat')

    K1 = np.array(K1['K1'])
    K2 = np.array(K2['K2'])
    D1 = np.array(D1['D1'])
    D2 = np.array(D2['D2'])
    R = np.array(R['R'])
    t = np.array([t['T']])

    t = np.reshape(np.ravel(t), (3, 1))   # added by manan
      
    return K1.T, K2.T, D1, D2, R, t 


def es_extract_orb_pairs(img_l, img_r) :
    """
    ** Not used in the pipeline **
    Description:
        This function finds matching point pairs 
        in two images observing the same scene
    Args:
        img_l:
        img_r:
    Returns:
        pts_l : keypoints in the left image
        pts_r : corresponding matched keypoints in the right image
    """
    # Convert input images to grayscale
    img_l_bw = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
    img_r_bw = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # We shall use ORB features to detect keypoints in the left and right images
    # Initialize the ORB detector algorithm
    orb = cv2.ORB_create()
    
    # Now detect the keypoints and compute
    # the descriptors for the left image
    # and right image
    l_kp, des_l  = orb.detectAndCompute(img_l_bw,None)
    r_kp, des_r  = orb.detectAndCompute(img_r_bw,None)
    
    # Initialize the Matcher for matching
    # the keypoints and then match the
    # keypoints
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches = matcher.match(des_l,des_r)

    # Extract matched points
    pts_l = []
    pts_r = []
    thresh = 20
    for i,(m) in enumerate(matches):
        if m.distance < thresh : 
            #print(m.distance)
            pts_l.append(l_kp[m.trainIdx].pt)
            pts_r.append(r_kp[m.queryIdx].pt)
    pts_l = np.asarray(pts_l)
    pts_r = np.asarray(pts_r)
  
    es_draw_matches(img_l, img_r, pts_l, pts_r)

    return pts_l, pts_r


def es_undistort_image(img, K, D):
    """
    ** Not used in the pipeline **
    Description:
        This function undistorts an image
    Args:
        img : image to be undistorted
        K   : camera matrix
        D   : distortion coefficients
    Returns:
        undistorted_image : the undistorted and cropped image
    """
    h,  w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
    # undistort
    undistorted_image = cv2.undistort(img, K, D, None, K)
    # crop the image
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]
    print(new_K)
    # undistorted_image = cv2.flip(undistorted_image,0)

    return undistorted_image


def calculate_essential_matrix(img_l, img_r, K_l, K_r, D):
    """
    ** Not used in the pipeline **
    Description:
        Calculates the essential matrix given the camera
        intrinsics and fundamental matrix for the two 
        stereo pair (use numpy array)
    Args:    
        K_l: The camera intrinsics of left camera
        F:   The fundamental matrix
        K_r: The camera intrinsics of right camera
    Returns:
        E:   The essential matrix (numpy array)
    """
    # Set E to default value
    E = None

    # First undistort the image
    uimg_l = es_undistort_image(img_l,K_l, D)
    uimg_r = es_undistort_image(img_r,K_r, D)

    # Extract matching points
    pts_l, pts_r = es_extract_orb_pairs(uimg_l, uimg_r)

    method = cv2.RANSAC
    prob = 0.999 # probabilty
    threshold = 0.0003

    # E is estimated using Nisters 5 pt algorithm
    E, mask = cv2.findEssentialMat(pts_l, pts_r, K_l, method, prob, threshold)

    return E 


def calculate_pos_r(matches_l, matches_r, P_l, Mint_r):
    """
    * Not used in the pipeline **
    Description:
        This function calculates the x, y, z of the matches
        with respect to the right camera
    Args:
        matches_l: pixel coords (n x 2)
        matches_r: pixel coords (n x 2)
        Mint_r:    intrinsic matrix of the right camera with an extra column (3 x 4)
                    | fx   0  ox  0 |
                    |  0  fy  oy  0 | 
                    |  0   0   1  0 |
        P_l:       projection matrix of the right camera (4 x 4)
                    | r11  r12  r13  tx |
                    | r12  r22  r23  ty | 
                    | r31  r32  r33  tz |
                    |   0    0    0   1 |
    Returns:
        poses: poses of all the matches found (wrt the right camera) (n x 3)
    """
    poses = []
    assert matches_l.shape == matches_r.shape

    ##################################################
    m_vec = np.expand_dims(np.array([Mint_r[2, 0], Mint_r[2, 1], Mint_r[2, 2]]), axis=0)
    m_sub = np.concatenate((Mint_r[0:2, 0:3], P_l[0:2, 0:3]), axis=0)
    B     = np.array([[Mint_r[0, 3] - Mint_r[2, 3]], [Mint_r[1, 3] - Mint_r[2, 3]], [P_l[0, 3] - P_l[1, 3]], [P_l[1, 3] - P_l[2, 3]]])
    A     = None

    for pt_r, pt_l in zip(matches_l, matches_r):
        uv_vect = np.expand_dims(np.concatenate((pt_r, pt_l)).T, axis=1)
        A = uv_vect * m_vec - m_sub
        print("A = {}, B = {}".format(A.shape, B.shape))
        poses.append(np.linalg.solve(np.matmul(A.T, A), np.matmul(A.T, B)))

    poses = np.squeeze(np.array(poses).astype(np.float32), axis=2)
    ##################################################

    return poses


def svd_essential_matrix(E, matches_l, matches_r):
    """
    ** Not used in the pipeline **
    Description:
        Run svd to decompose the essential matrix into Tx (skew symmetric transform)
        and R (orthonormal rotation matrix). This produces 2 R matrices and 2 Tx matrices
        It then checks which of 4 solutions is correct

        Tx = |   0  -tz   ty |
             |  tz    0  -tx | 
             | -ty   tx    0 |
            
        R   = |  r11  r12   r13 |
              |  r21  r22   r23 | 
              |  r31  r32   r33 |
    Args:
        E: Essential matrix (numpy array)
        matches_l: matching keypoints from left  camera (nx2) (from find_matches function)
        matches_r: matching keypoints from right camera (nx2) (from find_matches function)
    Returns:
        Tx: transformation matrix (3x3) skew symmwtric
        R:   Rotation matrix (3x3)
    """

    # Make sure the arrays are all the same "float" type
    matches_l = matches_l.astype(np.float)
    matches_r = matches_r.astype(np.float)
    E = E.astype(np.float)

    # cv2.recoverPose Outputs: 
    # R(3x3), T(3x1), TriPoints - triangulated 3D points from pair of matching pixels
    TriPoints, R, T, _ = cv2.recoverPose(E, matches_l, matches_r)

    # Converts T into a 3x3 skew symmetric matrix
    Tx = np.array([0,-T[2],T[1],    T[2],0,-T[0],    -T[1],T[0],0]).reshape(3, 3)

    return Tx, R, TriPoints


def es_extract_sift_pairs(img_l, img_r):
    """
    ** Not used in the pipeline **
    """
    img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # sift
    sift = cv2.xfeatures2d.SIFT_create()

    kp_l, des_l = sift.detectAndCompute(img_l, None)
    kp_r, des_r = sift.detectAndCompute(img_r, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(des_l, des_r)
    matches = sorted(matches, key = lambda x:x.distance)

    matched_points = cv2.drawMatches(img_l, kp_l, img_r, kp_r, matches[:100], img_r, flags=2)
    cv2.imwrite("matched_points.jpg", matched_points)


def rectify_images(imageA, imageB):
    """
    ** Not used in the pipelin **
    """
    # convert to black and white
    img1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    #sif
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    # matches = sorted(matches, key = lambda x:x.distance)

    num_matches = min(1000, len(matches))
    # num_matches = len(matches)
    matches = matches[:num_matches]

    # get the x and y co-ordinates in both images
    matchesA = []
    matchesB = []

    # For each match...
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        x1, y1 = keypoints_1[img1_idx].pt
        x2, y2 = keypoints_2[img2_idx].pt

        matchesA.append([x1, y1])
        matchesB.append([x2, y2])

    matchesA = np.array(matchesA).astype(np.int32)
    matchesB = np.array(matchesB).astype(np.int32)

    fundamental_matrix, inliers = cv2.findFundamentalMat(matchesA, matchesB, cv2.FM_RANSAC, 1, 0.9)
    print(fundamental_matrix)

    # We select only inlier points
    pts1 = matchesA[inliers.ravel() == 1]
    pts2 = matchesB[inliers.ravel() == 1]

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
                np.float32(pts1), 
                np.float32(pts2), 
                fundamental_matrix, imgSize=(w1, h1))

    # RECTIFY IMAGES
    h1, w1 = (imageA.shape[0], imageA.shape[1])
    h2, w2 = (imageB.shape[0], imageB.shape[1])

    img1_rectified = cv2.warpPerspective(imageA, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(imageB, H2, (w2, h2))

    # generate the corrected and uncorrected perspectives
    h,w = img1_rectified.shape[:2]
    vis_c = np.zeros((h,w+w,3), dtype="uint8") # create a mosaic image
    vis_c[0:h,0:w] = img1_rectified
    vis_c[0:h,w:] = img2_rectified

    cv2.imwrite("corrected_perspective.jpg", vis_c)
        
    return img1_rectified, img2_rectified


def es_triangulate_point(proj_mat_l, proj_mat_r, pt_l, pt_r):
	"""
	Description:
	This function calculates the position of a point in 3d using
	the DLT (Direct Linear transform) algorithm
	Args:
	proj_mat_l: left camera projection matrix
	proj_mat_r: right camera projection matrix
	pt_l : point in the left camera
	pt_r : corresponding point in the right camera

	Returns:
	obj_position : 3D position of the object
	"""
	A = [ pt_l[1]*proj_mat_l[2,:]-proj_mat_l[1,:],
	proj_mat_l[0,:]-pt_l[0]*proj_mat_l[2,:],
	pt_r[1]*proj_mat_r[2,:]-proj_mat_r[1,:],
	proj_mat_r[0,:]-pt_r[0]*proj_mat_r[2,:]
	]
	A = np.array(A).reshape((4,4))
	B = np.matmul(A.T,A)
	U,s,Vh = linalg.svd(B, full_matrices=False)

	return Vh[3,0:3]/Vh[3,3]


def es_get_proj_mat(K,R,t):
    """
    Description : 
        This function computes the projection matrix (KRt) for a camera
        It requires provision of the camera matrix K, Rotation Matrix R and 
        Translation vector t
    P = K*[Rt]
    """
    Rt = np.concatenate([R, t], axis = -1)
    P = K @ Rt #projection matrix
    return P