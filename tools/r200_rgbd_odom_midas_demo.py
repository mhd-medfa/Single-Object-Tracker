from collections import deque
import rospy
from sensor_msgs.msg import CameraInfo, Image as ImageMsg
import message_filters
import cv2
import os
import numpy as np
# import ros_numpy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
import glob
from test import *
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from experiments.siammask_sharp.custom import Custom
from sort import *
from Bezier import Bezier
from collections import deque
from MiDaS import MiDaS
import copy

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
        self.moving_object_odom_rate = rospy.Rate(30.0)

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
        # self.image = ros_numpy.numpify(rgb_msg)
        self.image = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, -1)
        self.undist_image = cv2.undistort(self.image, self.camera_info_K, self.camera_info_D)
        # self.depth = ros_numpy.numpify(depth_msg)
        self.depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width, -1)
    
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


if __name__ == '__main__':
     # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    #create instance of SORT
    sort_tracker = Sort()
    
    # Load Kalman filter to predict the trajectory
    kf = KalmanFilter()
    
    # MiDaS
    midas = MiDaS()
    # Setup Model
    cfg = load_config(args)
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)
    
    rospy.init_node("siammaskimage", anonymous=True)
    # rospy.init_node('odometry_publisher')

    odom_pub = rospy.Publisher("moving_object_odom", Odometry, queue_size=50)
    odom_broadcaster = tf.TransformBroadcaster()
    
    my_node = Nodo()
    _ = my_node.frame_capture()
    frame = my_node.undist_frame_capture()
    depth_frame = my_node.depth_frame_capture()
    relative_depth_frame_colored = midas.estimate(frame)
    
    # Select ROI
    cv2.namedWindow("Demo", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('Demo', frame, False, False)
        x, y, w, h = init_rect
    except Exception as e:
        print(e)
        print("Something wrong happened!")
        exit()

    toc = 0
    f = 0
    camera_focal_length_x = my_node.camera_info_K[0,0] #fx
    camera_focal_length_y = my_node.camera_info_K[1,1] #fy
    camera_principle_point_x = my_node.camera_info_K[0,2] #x0
    camera_principle_point_y = my_node.camera_info_K[1,2] #y0
    
    current_time = rospy.Time.now()
    last_time = rospy.Time.now()
    position_queue = deque(maxlen=6)
    velocity_queue = deque(maxlen=6)
    t_points = np.arange(0, 1, 0.01)
    # depth_hybrid = np.zeros_like(depth_frame)
    # create an inverse from the colormap to gray values
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_HOT).reshape(256, 3))
    color_to_gray_map = dict(zip(color_values, gray_values))
    s=True
    while not rospy.is_shutdown():
        current_time = rospy.Time.now()
        # since all odometry is 6DOF we'll need a quaternion created from yaw
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, 0)
        
        tic = cv2.getTickCount()
        # Capture the video frame
        # by frame
        _ = my_node.frame_capture()
        frame = my_node.undist_frame_capture()
        depth_frame = my_node.depth_frame_capture()
        relative_depth_frame, magma_relative_depth_map = midas.estimate(frame)
        # relative_depth_frame = cv2.cvtColor(relative_depth_frame_colored, cv2.COLOR_RGB2GRAY)
        # relative_depth_frame.reshape(relative_depth_frame.shape[0], relative_depth_frame.shape[1], 1)
        relative_depth_frame = relative_depth_frame[..., np.newaxis].astype(np.float)
        # cv2.imshow('undist_SiamMask', undist_frame)
        # cv2.imshow('depth_SiamMask', depth_frame)
        # cv2.imshow("relative_depth_frame_colored", relative_depth_frame_colored)
        
        
        depth_masked = copy.deepcopy(depth_frame)
        depth_masked[np.isnan(depth_masked)] = 0
        inversed_relative_depth_frame = (1. - relative_depth_frame)
        normalized_inversed_relative_depth_frame = (inversed_relative_depth_frame - np.min(inversed_relative_depth_frame))/(np.max(inversed_relative_depth_frame)-np.min(inversed_relative_depth_frame))
        depth_ratio_array = depth_masked / (normalized_inversed_relative_depth_frame+1e-5)
        depth_ratio = depth_ratio_array[np.nonzero(depth_ratio_array)].mean()
        depth_ratio_std = depth_ratio_array[np.nonzero(depth_ratio_array)].std()
        print("DePtH RaTi0 !$$$$$$$$$$$$$$$$$$$$$$$$:")
        print(depth_ratio)
        print("$$$$$$$$Td:")
        print(depth_ratio_std)
        # depth_frame = copy.deepcopy(relative_depth_frame)
        # depth_temp = copy.deepcopy(depth_frame)
        # depth_temp = depth_frame
        # depth_temp = np.where(np.isnan(depth_temp), inversed_relative_depth_frame*depth_ratio, depth_temp)
        depth_temp = normalized_inversed_relative_depth_frame*depth_ratio
        depth_hybrid = depth_temp
        cv2.imshow("depth", depth_frame)
        depth_frame = depth_hybrid
        
        cv2.imshow("relative_depth", magma_relative_depth_map)
        cv2.imshow("test", (255*relative_depth_frame).astype(np.uint8))
        cv2.imshow("depth_hybrid", depth_hybrid)
        # import pandas as pd 
        # if s == True:
        #     s=False
        #     pd.DataFrame(np.squeeze(depth_masked)).to_csv("file.csv")
        #     pd.DataFrame(np.squeeze(relative_depth_frame)).to_csv("file2.csv")
            
        # cv2.waitKey(0)
        
        if f == 0:  # init
            f=1
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(frame, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
            print(depth_frame)
            print("relative depth")
            print( relative_depth_frame)
        elif f > 0:  # tracking
            state = siamese_track(state, frame, sort_tracker=sort_tracker, mask_enable=True, refine_enable=True, device=device)  # track
            if state['score'] < 0.65:
                pass #here should call panoptic segmentation on the latest good frame with high score
            elif state['score'] >= 0.65:
                high_score_frame = frame
            else:
                print("Check state's score value")
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr
            predicted = kf.predict(state['target_pos'][0], state['target_pos'][1])
            #cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 4)
            cv2.circle(frame, (int(state['target_pos'][0]), int(state['target_pos'][1])), 20, (0, 0, 255), 4)
            cv2.circle(frame, (int(predicted[0]), int(predicted[1])), 20, (255, 0, 0), 4)
            x_image = int(state['target_pos'][0])
            y_image = int(state['target_pos'][1])
            z_3D = depth_frame[y_image, x_image][0]
            
            x_3D, y_3D, z_3D = convert_2D_to_3D_coords(x_image=x_image, y_image=y_image, x0=camera_principle_point_x, y0=camera_principle_point_x,
                                    fx=camera_focal_length_x, fy=camera_focal_length_y, z_3D=z_3D)
            position_queue.append([z_3D, x_3D, y_3D])
            if f==1:
                x_3D_old, y_3D_old, z_3D_old = x_3D, y_3D, z_3D
            vx_3D, vy_3D, vz_3D = x_3D - x_3D_old, y_3D - y_3D_old, z_3D - z_3D_old
            velocity_queue.append([vz_3D, vx_3D, vy_3D])
            print("X-target = {}, Y-target = {}, Z-target = {}".format(x_3D, y_3D, z_3D))
            
            # first, we'll publish the transform over tf
            odom_broadcaster.sendTransform(
                (z_3D, y_3D, x_3D),
                odom_quat,
                current_time,
                "base_link",
                "moving_object_odom"
            )
            
            # next, we'll publish the odometry message over ROS
            odom = Odometry()
            odom.header.stamp = current_time
            odom.header.frame_id = "moving_object_odom"
            curve_position_set = Bezier.Curve(t_points, np.array(position_queue))
            curve_velocity_set = Bezier.Curve(t_points, np.array(velocity_queue))
            #publish bezier curves of position and velocity
            for pos, vel in zip(curve_position_set, curve_velocity_set):
                # set the position
                odom.pose.pose = Pose(Point(*pos), Quaternion(*odom_quat))

                # set the velocity
                odom.child_frame_id = "base_link"
                odom.twist.twist = Twist(Vector3(*vel), Vector3(0, 0, 0))

                # publish the message
                odom_pub.publish(odom)

            last_time = current_time
            x_3D_old, y_3D_old, z_3D_old = x_3D, y_3D, z_3D
            my_node.moving_object_odom_rate.sleep()
            
            frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
            cv2.polylines(frame, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            if state['track_bbs_ids'].size>0:
                x1 = int(state['track_bbs_ids'][-1][0])
                y1 = int(state['track_bbs_ids'][-1][1])
                x2 = int(state['track_bbs_ids'][-1][2])
                y2 = int(state['track_bbs_ids'][-1][3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.imshow('Demo', frame)
            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
    