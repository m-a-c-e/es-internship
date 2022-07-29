import cv2
import numpy as np  
import zmq
import threading
from   blob_detector_helpers import *
from   depth_from_stereo_helpers import *
import time
from   double_thread import double_thread, six_thread


class VideoCapture:
    
    def __init__(self, name) :
        self.cap = cv2.VideoCapture(name,cv2.CAP_GSTREAMER)
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # consume frames as they come, keep only most recent
    def _reader(self):
        while True : 
            ret = self.cap.grab()
            if not ret:
                break
    #retrieve latest frame
    def read(self):
        ret, frame = self.cap.retrieve()
        
        return frame
        #return self.q.get()


def main():
    iterator     = 0
    context      = None
    sub_socket   = None
    img_l_color  = None
    img_r_color  = None
    server       = False
    stream       = False
    display_on   = False
    write_flag   = False
    write_single = False
    list_r1      = []
    list_r2      = []
    list_r3      = []
    list_r4      = []
    list_total_rt= []
    print_stats  = False

    win_size = 224              # for sift descriptor (preferably should be divisible by 4)

    name_right_cam = 'udpsrc port=50002 do-timestamp=1 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)JPEG, payload=(int)26" ! rtpjitterbuffer mode=0 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink'
    name_left_cam  = 'udpsrc port=50006 do-timestamp=1 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)JPEG, payload=(int)26" ! rtpjitterbuffer mode=0 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink'

    # NB : The width and height here are not for a single frame but rather are the size of the mosaic
    # 960 x 1280 is orlacos resolution which is being used
    frame_width = win_size + (1280 * 2)
    frame_height = win_size + (960 * 2)
    frame_size = (frame_width, frame_height)
    vid_writer = cv2.VideoWriter('results/output_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, frame_size)

    # changes with images
    # dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/tl_data_07_13_2022/'     --> calibration is off
    # dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/07_20_2022/orlaco/comp_zed_cam/'
    # dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/07_20_2022/orlaco/comp_zed_cam_v2/'
    # dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/07_21_2022/stream_data_orlaco_v2/'
    # dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/07_22_2022/stream_data_orlaco_v3/'
    dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/cloudy_data/cloudy_day_orlaco/cloudy_day_orlaco_v1/'
    # dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/cloudy_data/cloudy_day_orlaco/cloudy_day_orlaco_v2/'

    # dataset_path = '/Users/manan.patelequipmentshare.com/Desktop/last_dataset_chair/detection_results_chair_orlaco/'
    # dataset_path = 'Stereovision_input_data/07_20_2022/tl_data_o1/'

    # get the image height and width
    dummy_img = cv2.imread(dataset_path + 'left-0.jpg')
    height = dummy_img.shape[0]
    width  = dummy_img.shape[1]

    # initialize the source
    context    = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect('tcp://127.0.0.1:{}'.format(9001))
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')
    leftBytes  = None
    rightBytes = None
    right_cam = VideoCapture(name_right_cam)
    left_cam = VideoCapture(name_left_cam)

    #______________________________Get Projection Matrices______________________________
    K1, K2, D1, D2, R, t = read_calib_files(dataset_path + 'cam-config/')
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    P1 = es_get_proj_mat(K1,R1,t1)
    R  = R.T
    R2 = copy.copy(R)
    t2 = copy.copy(t)
    P2 = es_get_proj_mat(K2, R2, t2)

    while(True):
        if iterator == 100:
            break
        key = cv2.waitKey(1)&0xFF
        st = time.time()
        if server:
            leftBytes = sub_socket.recv_multipart()[1]
            rightBytes = sub_socket.recv_multipart()[1]
            buf = np.ndarray(shape=(1, len(leftBytes)), dtype=np.uint8, buffer=leftBytes)
            img_l_color = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            buf = np.ndarray(shape=(1, len(rightBytes)), dtype=np.uint8, buffer=rightBytes)
            img_r_color = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        elif stream : # In which case, input data is coming in from the two orlaco cameras.
                      # Herein, we shall `poll` the threads handling each cameras streams
                      # for a single frame
            #print("Reading image frames from Videocapture stream...")
            img_r_color = right_cam.read() # right cam is port 50002
            img_l_color = left_cam.read() # left cam is port 50006
            if( (img_r_color is None) or (img_l_color is None ) ) :
                print("Frames have not arrived yet")
                continue
        else:
            #______________________________Initialization______________________________
            img_l_path = dataset_path + 'left-' + str(iterator) + '.jpg'
            img_r_path = dataset_path + 'right-' + str(iterator) + '.jpg'
            img_l_color = cv2.imread(img_l_path)        # <------ 6 ms
            img_r_color = cv2.imread(img_r_path)        # <------ 6 ms

        r1 = time.time() - st

       #______________________________Image processing______________________________
        st = time.time()

        # undistort (uncomment if providing distorted images)           
        # --Multithread--
        img_l_color, img_r_color = double_thread( cv2.undistort, 
            [ img_l_color, K1, np.array([-0.25, 0.05, 0, 0]) ],         # -0.25 and 0.05 are distortion coefficients which are
            [ img_r_color, K2, np.array([-0.25, 0.05, 0, 0]) ])         # hardcoded for orlacos after making an educated guess from 
                                                                        # calibrated values. comment out if using zed images

        # pad images
        ############################# Threads make this slower! ###########################
        img_l_color = cv2.copyMakeBorder(img_l_color, win_size // 2, win_size // 2, win_size // 2, win_size // 2, cv2.BORDER_CONSTANT, value=[100, 100, 100])
        img_r_color = cv2.copyMakeBorder(img_r_color, win_size // 2, win_size // 2, win_size // 2, win_size // 2, cv2.BORDER_CONSTANT, value=[100, 100, 100])
               
        # Convert to hsv
        img_l_color_hsv = cv2.cvtColor(img_l_color, cv2.COLOR_BGR2HSV)
        img_r_color_hsv = cv2.cvtColor(img_r_color, cv2.COLOR_BGR2HSV)
        
        # increase saturation
        img_l_color_hsv = inc_sat(img_l_color_hsv, 1.1)
        img_r_color_hsv = inc_sat(img_r_color_hsv, 1.1)

        # ______________________________Correspondence_________________________________
        # color filtering (returns the grayscale blobs image)
        img_l_new = copy.copy(img_l_color)
        img_r_new = copy.copy(img_r_color)
        l_all_points = []
        r_all_points = []
        r2 = time.time() - st
        st = time.time()
        l_coords, l_boxes, res_l, r_coords, r_boxes, res_r = six_thread(complete_blob,
                                                            [img_l_color_hsv, 18, 20, 33, 40, 5, 10],
                                                            [img_r_color_hsv, 18, 20, 33, 40, 5, 10])

        r2 = time.time() - st

        # draw bounding boxes
        for i in range(len(l_boxes)):
            cv2.rectangle(img_l_new,(l_boxes[i][0], l_boxes[i][1]),(l_boxes[i][2], l_boxes[i][3]),(255,0,0), 2)

        for i in range(len(r_boxes)):
            cv2.rectangle(img_r_new,(r_boxes[i][0], r_boxes[i][1]),(r_boxes[i][2], r_boxes[i][3]),(255,0,0), 2)

        # generate keypoints                                            

        st = time.time()
        # finding matches between keypoints using SIFT
        l_feats = generate_features(img_l_color, l_coords, win_size)
        r_feats = generate_features(img_r_color, r_coords, win_size)

        matches = match_features(l_feats, r_feats)

        if len(matches) == 0:
            pass
        elif len(l_all_points) == 0 or len(r_all_points) == 0:
            matches_l_sift = l_coords[matches[:, 0]]
            matches_r_sift = r_coords[matches[:, 1]]
            matches_l_sift -= np.array(win_size // 2)
            matches_r_sift -= np.array(win_size // 2)
            l_all_points = copy.copy(matches_l_sift)
            r_all_points = copy.copy(matches_r_sift)
        else:
            matches_l_sift = l_coords[matches[:, 0]]
            matches_r_sift = r_coords[matches[:, 1]]
            matches_l_sift -= np.array(win_size // 2)
            matches_r_sift -= np.array(win_size // 2)
            l_all_points = np.concatenate((l_all_points, matches_l_sift), axis=0)
            r_all_points = np.concatenate((r_all_points, matches_r_sift), axis=0)
        
        r3 = time.time() - st
        #______________________________Depth______________________________
        st = time.time()
        Z = []
        X = []

        for u, v in zip(l_all_points, r_all_points):
            Z.append(es_triangulate_point(P1, P2, u, v)[2])
            X.append(es_triangulate_point(P1, P2, u, v)[0])
            
        Z = np.array(Z).astype(np.float32) * (1/1000)    # conversion to meters
        X = np.array(X).astype(np.float32) * (1/1000)    # conversion to meters
        radial_depth = np.sqrt(np.square(Z) + np.square(X)) # in meters
        r4 = time.time() - st

        if len(l_all_points) != 0 or len(r_all_points) != 0:
            l_all_points += np.array(win_size // 2)
            r_all_points += np.array(win_size // 2)

        matched_points_sift       = es_draw_matches(img_l_new, img_r_new, l_all_points, r_all_points, None, None, None)
        matched_points_sift_depth = es_draw_matches(img_l_new, img_r_new, l_all_points, r_all_points, None, None, radial_depth)

        #______________________________Results______________________________
        res_l    = np.stack((res_l,)*3, axis=-1)
        res_r    = np.stack((res_r,)*3, axis=-1)

        m1     = np.concatenate((res_l, res_r), axis=1)
        m2     = np.concatenate((img_l_new, img_r_new), axis=1)
        mosaic = np.concatenate((m1, m2), axis=0)
        mosaic = np.concatenate((mosaic, matched_points_sift), axis=0)
        mosaic = np.concatenate((mosaic, matched_points_sift_depth), axis=0)

        total_r = r1 + r2 + r3 + r4
        if(print_stats):
            print("Total runtime = {} ms".format(np.round(total_r * 1000)))
            print("%Percentages%")
            print("Init images: {} ({}) %".format(np.round(r1 * 1000, 2), np.round(r1 * 100 / total_r, 2)))
            print("Blob detection: {} ({}) %".format(np.round(r2 * 1000, 2), np.round(r2 * 100 / total_r, 2)))
            print("Sift: {} ({}) %".format(np.round(r3 * 1000, 2), np.round(r3 * 100 / total_r, 2)))
            print("Depth: {} ({}) %".format(np.round(r4 * 1000, 2), np.round(r4 * 100 / total_r, 2)))
            print()

        if not server and not stream and not display_on:
            cv2.imwrite(dataset_path + 'results/' + 'mosaic' + str(iterator) + '.jpg', mosaic)
            cv2.imwrite('mosaic' + str(iterator) + '.jpg', mosaic[(height + win_size)*3:, :(width + win_size), :])
            if write_single:
                break
        
        if display_on :
            iterator += 9
            cv2.imshow('Display_Image', mosaic[(height + win_size)*3:, :(width + win_size), :])
            cv2.waitKey(1)

        # only write video frames at 10Hz -- based on iterator
        if(write_flag) :
            if( (iterator % 10) == 0 ) :
                vid_writer.write(mosaic[(height + win_size)*3:, :(width + win_size), :])

        # collect timing stats for each iteration of the loop
        list_r1.append(r1)
        list_r2.append(r2)
        list_r3.append(r3)
        list_r4.append(r4)
        list_total_rt.append(total_r)

        if(key == ord("q") ):
            break
    
        if not(iterator % 100):
            print(iterator)

        iterator += 1
    #______________________________Saving time results______________________________
    save_path = dataset_path + '/results/'
    np.save(save_path + 'r1.npy',list_r1)
    np.save(save_path + 'r2.npy',list_r2)
    np.save(save_path + 'r3.npy',list_r3)
    np.save(save_path + 'r4.npy',list_r4)
    np.save(save_path + 'total_runtime.npy', list_total_rt)


if __name__ == "__main__":
    main()