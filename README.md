# Depth From Stereo Pipeline (Orlaco Cameras)

Given: a calibrated pair of orlaco cameras

- Pipeline input: &ensp;&ensp;&ensp;two stereo images
- Pipeline output: &ensp; one image marking the distance (meters) of safety vest colored objects (orange and yellow)

The mosaic image below shows the four major steps involved in the pipeline. These steps are discussed in detail below.
![Result](dfs.jpg?raw=true)
_Figure 1. The Four Major Steps Involved in the Pipeline_

## 1. Image Processing
&ensp;&ensp;&ensp;&ensp;&ensp;
 To detect the safety vest colours, the `cv2.inRange` function is used to keep the colors which are inside a predetermined range of HSV color values (see the `color_mask` function in `blob_detector_helpers.py`). This function creates a mask which can then be applied to the original image to remove the colors outisde this range. 

&ensp;&ensp;&ensp;&ensp;&ensp;
 One color filter is used for the neon green color and two are used for the orange color. The process of specifying the value range for each hue is simple when utilizing a flat hsv color map. [More details on how HSV color space works](https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/)

&ensp;&ensp;&ensp;&ensp;&ensp;
 However, the saturation and brightness changes with varying lighting conditions. When taking images in low light, these values need to be set to something lower than usual (roughly 100-150) so as to pick up colors. This might be one area where the algorithm can be improved upon to select the saturation and brightness depending on lighting conditions automatically.

&ensp;&ensp;&ensp;&ensp;&ensp;
 In order to reduce the noise after color filtering, a sequence of `cv2.erode` and `cv2.dilate` functions are used to clean up the image.

## 2. Blob Detection
&ensp;&ensp;&ensp;&ensp;&ensp;
 Once all the unncessary color ranges are filtered out, we create an instance of the `SimpleBlobDetector` class provided by opecv and specify the parameters needed. [More details about required parameters](https://learnopencv.com/blob-detection-using-opencv-python-c/). 

&ensp;&ensp;&ensp;&ensp;&ensp;
 We then pass into the blob detector the grayscale filtered image. The code then finds the centroids of deteted blobs based on given parameters, such as minimal size and other threshholds, that are provided from the initalization of the blob detector. One of the most important parameters here is the pixel area of the blobs. Since we do not want want blobs with area less than 200 pixels, we set `minArea` to 200 and `maxArea` to an arbitrarily high value, to further ignore the noise in our piepline. The second set of images in Figure 1 shows the centroids of blobs deteced.

&ensp;&ensp;&ensp;&ensp;&ensp;
 The collection of blob centroids for a given image is then passed on as a set of "keypoints" to the next part of the pipeline.

## 3. Feature Matching
&ensp;&ensp;&ensp;&ensp;&ensp;
 Once a set of "keypoints" is detected in both the images, the next step is to figure out which blobs in the left image correspond to the ones in the rigth image. This is then used to get the disparity between images which can then be used to compute depth. We use sift descriptors to get these matches. Around each keypoint, we take a `win_size` x `win_size` crop of the image (100 x 100 pixels in our case) to generate a feature vector for each of the keypoints. All the features are then normalized between 0 and 1. Next, we compute the euclidean distance between each feature descriptor in the left image and the right image to get a matrix of "distances". Since the features were initially normalized, these distances will also lie between 0 and 1. 

&ensp;&ensp;&ensp;&ensp;&ensp;
 To perform one to one matching, we extract the pair of descriptors which are the most similiar to each other (minimum distances) and fall below a given threshold value (0.5 in our case). These descriptors are then matched together. Once they are matched, we remove them from the keypoint lists of both images so that we only get one to one matches. The third set of images in Figure 1 shows the feature matching.

## 4. Triangulating Matches
&ensp;&ensp;&ensp;&ensp;&ensp;
 At this stage in the pipeline, we are ready to begin estimating 3D points. This is known as point triangulation and it can be done using a technique called the Direct Linear Transform or DLT. We use the DLT algorithm to estimate the 3D position of a 2D point given that: 1) We have a pair of matched 2D points i.e. (`u_cam1`,`v_cam1`) and corresponding (`u_cam2`,`v_cam2`). 2). We have the camera projection matrix `[K | R t]` of each camera. Where `K` is the camera's intrinsic matrix (focal length, principal point and skew) gotten from the camera calibration step. `R` and `t` are the rotation matrix and translation vector relating the two cameras which are also gotten from the camera calibration step. Following the pinhole camera model, we can relate a point in the real world (3D) to the image plane of a camera using the projection matrix. This geometrical relationship has the form `x=PX` where `x` is the point in 2D and `P` is the projection matrix of the camera and `X` is the 3D point. Using all of the information above, we form an overdetermined (2 cams) system of linear equations and solve for the unknown 3D point `X` using singular value decomposition.

## 5. Multithreading
&ensp;&ensp;&ensp;&ensp;&ensp;
 To speed up certain sections of the code, threads were used to run the same funtions for both camera inputs at the same time. Threading only consistently benifited the runtime from functions like `cv2.undistort` and `blob_detector`. The `double_thread` function automoactically runs the same function for two different sets of input arguments and then proceeds to wait for both threads to join before moving on. Inside that function is a subclass of threads, `CustomThread`, that is used so that the created threads can output the return value(s) of the given function. Additionally it was attempted (unsuccessfully) to decrease runtime by doing all the image processing functions for left and right camera images in single threads, as opposed to doing threads function by function.

