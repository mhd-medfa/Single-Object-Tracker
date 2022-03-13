import rospy
from sensor_msgs.msg import CameraInfo, Image as ImageMsg
import message_filters
import cv2
import os
import numpy as np
import ros_numpy
import pandas as pd
import glob
from test import *
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from experiments.siammask_sharp.custom import Custom
from sort import *

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='experiments/siammask_sharp/SiamMask_DAVIS.pth', type=str,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='experiments/siammask_sharp/config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

class Nodo(object):
    def __init__(self):
        # Params
        self.image = None
        self.undist_image = None
        self.depth = None
        self.camera_info_K = None
        self.camera_info_D = None

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(30)

        # Publishers
        # self.pub = rospy.Publisher('imagetimer', Image,queue_size=10)

        # Subscribers
        rgb_sub = message_filters.Subscriber("/r200/rgb/image_raw",ImageMsg)
        depth_sub = message_filters.Subscriber("/r200/depth/image_raw",ImageMsg)
        camera_info_msg = rospy.wait_for_message("/r200/rgb/camera_info",CameraInfo)#, self.info_callback)
        self.camera_info_K = np.array(camera_info_msg.K).reshape([3, 3])
        self.camera_info_D = np.array(camera_info_msg.D)
        ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub], 60)#, 0.2,allow_headerless=False)
        ts.registerCallback(self.callback)

    def callback(self, rgb_msg, depth_msg):
        self.image = ros_numpy.numpify(rgb_msg)
        self.undist_image = cv2.undistort(self.image, self.camera_info_K, self.camera_info_D)
        self.depth = ros_numpy.numpify(depth_msg)
    
    def frame_capture(self):
        self.loop_rate.sleep()
        return self.image

    def undist_frame_capture(self):
        self.loop_rate.sleep()
        return self.undist_image

    def depth_frame_capture(self):
        self.loop_rate.sleep()
        return self.depth

class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y
def convert_2D_to_3D_coords(x_image, y_image, x0, y0, fx, fy, z_3D):
    """
    you can find the values of the camera intrinsic parameters at ./data/depth_Depth_metadata.csv
    """
    
    camera_principle_point_x = x0
    camera_principle_point_y = y0
    camera_focal_length_x = fx
    camera_focal_legnth_y = fy

    # Formuals to calculate the x and y in 3D (As we studied Pinhole camera model in the lab )
    x_3D = (x_image - camera_principle_point_x) * z_3D / camera_focal_length_x
    y_3D = (y_image - camera_principle_point_y) * z_3D / camera_focal_legnth_y
    
    return x_3D, y_3D, z_3D

def _4EMA(df):
    # avg_gain=df.gain.ewm(span=12,min_periods=12,adjust=False).mean()
    EMA9 = df.ewm(span=9,min_periods=9,adjust=False).mean()
    EMA13 = df.ewm(span=13,min_periods=13,adjust=False).mean()
    EMA21 = df.ewm(span=21,min_periods=21,adjust=False).mean()
    EMA55 = df.ewm(span=55,min_periods=55,adjust=False).mean()
    return EMA9, EMA13, EMA21, EMA55

if __name__ == '__main__':
     # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    #create instance of SORT
    sort_tracker = Sort()
    
    # Load Kalman filter to predict the trajectory
    kf = KalmanFilter()
    
    # Setup Model
    cfg = load_config(args)
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)
    
    rospy.init_node("siammaskimage", anonymous=True)
    my_node = Nodo()
    _ = my_node.frame_capture()
    frame = my_node.undist_frame_capture()
    depth_frame = my_node.depth_frame_capture()
    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', frame, False, False)
        x, y, w, h = init_rect
    except:
        print("Something wrong happened!")
        exit()

    toc = 0
    f = 0
    camera_focal_length_x = my_node.camera_info_K[0,0] #fx
    camera_focal_length_y = my_node.camera_info_K[1,1] #fy
    camera_principle_point_x = my_node.camera_info_K[0,2] #x0
    camera_principle_point_y = my_node.camera_info_K[1,2] #y0
    trajectory_df = pd.DataFrame()
    while not rospy.is_shutdown():
        tic = cv2.getTickCount()
        # Capture the video frame
        # by frame
        _ = my_node.frame_capture()
        frame = my_node.undist_frame_capture()
        depth_frame = my_node.depth_frame_capture()
        # cv2.imshow('undist_SiamMask', undist_frame)
        # cv2.imshow('depth_SiamMask', depth_frame)
        if f == 0:  # init
            f=1
            target_pos = np.array([x + w / 2, y + h / 2], dtype=np.int)
            target_sz = np.array([w, h])
            state = siamese_init(frame, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
            traj_step = pd.Series(target_pos, index=["x", "y"])
            trajectory_df = pd.concat([trajectory_df, traj_step.to_frame().T], ignore_index=True)
        
        elif f > 0:  # tracking
            state = siamese_track(state, frame, sort_tracker=sort_tracker, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr
            predicted = kf.predict(state['target_pos'][0], state['target_pos'][1])
            #cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 4)
            cv2.circle(frame, (int(state['target_pos'][0]), int(state['target_pos'][1])), 20, (0, 0, 255), 4)
            cv2.circle(frame, (int(predicted[0]), int(predicted[1])), 20, (255, 0, 0), 4)
            x_image = int(state['target_pos'][0])
            y_image = int(state['target_pos'][1])
            z_3D = depth_frame[y_image, x_image]
            
            traj_step = pd.Series([x_image, y_image], index=["x", "y"])
            trajectory_df = pd.concat([trajectory_df, traj_step.to_frame().T], ignore_index=True)
            if len(trajectory_df)>=55:
                EMA9, EMA13, EMA21, EMA55 = _4EMA(trajectory_df)
            while len(trajectory_df) > 60:
                n = 60 - len(trajectory_df)
                trajectory_df.drop(trajectory_df.tail(n).index,inplace=True) # drop last n rows
            
            x_3D, y_3D, z_3D = convert_2D_to_3D_coords(x_image=x_image, y_image=y_image, x0=camera_principle_point_x, y0=camera_principle_point_x,
                                    fx=camera_focal_length_x, fy=camera_focal_length_y, z_3D=z_3D)
            print("X-target = {}, Y-target = {}, Z-target = {}".format(x_3D, y_3D, z_3D))
            
            frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
            cv2.polylines(frame, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            if state['track_bbs_ids'].size>0:
                x1 = int(state['track_bbs_ids'][-1][0])
                y1 = int(state['track_bbs_ids'][-1][1])
                x2 = int(state['track_bbs_ids'][-1][2])
                y2 = int(state['track_bbs_ids'][-1][3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.imshow('SiamMask', frame)
            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))

# rospy.init_node("siammaskimage", anonymous=True)
# my_node = Nodo()
# frame = my_node.frame_capture()

# while not rospy.is_shutdown():
#     print("capturing..")
#     frame = my_node.frame_capture()
#     cv2.imshow('SiamMask', frame)
#     key = cv2.waitKey(1)
#     print(key)
#     if key > 0:
#         print("done")
#         break
#     # k = cv2.waitKey(1) & 0xFF
#     # if k == 27:
#     #     break