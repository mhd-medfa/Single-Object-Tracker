from collections import deque
import cv2
import os
import numpy as np
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
parser.add_argument('--base_path', default='data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

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

class KalmanFilterEstimator:
    # Parameters Initialization
    def __init__(self, sensor_mean, sensor_variance, sensor_reading) -> None:
        self.mu_xi, self.var_xi = 0.0, 6400 # mean and variance of the model
        self.mu_eta, self.var_eta = sensor_mean, sensor_variance # mean and variance of the sensor
        self.x_opt = 0.0  # optimal filtered values
        self.e = 0.0 # mean of the square errors
        self.K = 0.0 # Kalman coefficient's value over time

        # base of the iteration
        self.x_opt = 0.0
        self.e = self.var_eta
        self.base_iteration = True
        
    def step(self, sensor_reading):
        z = sensor_reading # sensor readings
        if self.base_iteration:
            self.base_iteration = False
            self.x_opt = z
        self.e = self.var_eta*(self.e+self.var_xi)/(self.e+self.var_xi+self.var_eta)
        self.K = self.e/self.var_eta
        self.x_opt = self.K*z + (1-self.K)*(self.x_opt)
        return self.x_opt

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
    
    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]
    
    # Select ROI
    cv2.namedWindow("Demo", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('Demo', ims[0], False, False)
        x, y, w, h = init_rect
    except Exception as e:
        print(e)
        print("Something wrong happened!")
        exit()

    toc = 0
    f = 0
    position_queue = deque(maxlen=6)
    velocity_queue = deque(maxlen=6)
    t_points = np.arange(0, 1, 0.01)
    # depth_hybrid = np.zeros_like(depth_frame)
    # create an inverse from the colormap to gray values
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_HOT).reshape(256, 3))
    color_to_gray_map = dict(zip(color_values, gray_values))
    
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        relative_depth_frame, magma_relative_depth_map = midas.estimate(im)
        frame = copy.deepcopy(im)
        relative_depth_frame = relative_depth_frame[..., np.newaxis].astype(np.float)
        inversed_relative_depth_frame = (1. - relative_depth_frame)
        inversed_relative_depth_frame_std = inversed_relative_depth_frame.std()
        inversed_relative_depth_frame_mean = inversed_relative_depth_frame.mean()
        depth_im = (255*inversed_relative_depth_frame).astype(np.uint8)
        
        cv2.imshow("relative_depth", magma_relative_depth_map)
        cv2.imshow("depth_im", depth_im)
        
        if f == 0:  # init
            # f=1
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            x_image = int(target_pos[0])
            y_image = int(target_pos[1])
            z_3D = depth_im[y_image, x_image][0]
            kf_estimator = KalmanFilterEstimator(inversed_relative_depth_frame_mean, inversed_relative_depth_frame_std**2, z_3D)
            target_depth = z_3D = int(kf_estimator.step(z_3D))
            state = siamese_init(im, target_pos, target_sz, target_depth, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, depth_im, siammask, cfg, sort_tracker=sort_tracker, mask_enable=True, refine_enable=True, reset_template=True, device=device)  # track
            if state['score'] < 0.65:
                pass #here should call panoptic segmentation on the latest good frame with high score
            elif state['score'] >= 0.65:
                high_score_frame = im
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
            z_3D = depth_im[y_image, x_image][0]
            z_3D = kf_estimator.step(z_3D)
            position_queue.append([z_3D, None, None])
            if f==1:
                z_3D_old = z_3D
            vz_3D = z_3D - z_3D_old
            velocity_queue.append([vz_3D, None, None])
            print("Z-target = {}".format(z_3D))

            z_3D_old = z_3D
            
            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
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
    