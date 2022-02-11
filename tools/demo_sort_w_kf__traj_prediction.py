# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from test import *
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from experiments.siammask_sharp.custom import Custom
# from kalmanfilter import KalmanFilter
from sort import *

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

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

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

    #create instance of SORT
    sort_tracker = Sort()
    
    # Load Kalman filter to predict the trajectory
    kf = KalmanFilter()
    
    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect
    except:
        exit()

    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, sort_tracker=sort_tracker, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            print("target position frame_id_{}:".format(f))
            print(state['target_pos'])
            print("target size frame_id_{}:".format(f))
            print(state['target_sz'])
            print("target dets frame_id_{}:".format(f))
            print(state['dets'])
            print("target track_bbs_ids frame_id_{}:".format(f))
            print(state['track_bbs_ids'])
            print("score frame_id_{}:".format(f))
            print(state['score'])
            
            mask = state['mask'] > state['p'].seg_thr
            predicted = kf.predict(state['target_pos'][0], state['target_pos'][1])
            #cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 4)
            cv2.circle(im, (int(state['target_pos'][0]), int(state['target_pos'][1])), 20, (0, 0, 255), 4)
            cv2.circle(im, (int(predicted[0]), int(predicted[1])), 20, (255, 0, 0), 4)
            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            
            if state['track_bbs_ids'].size>0:
                x1 = int(state['track_bbs_ids'][-1][0])
                y1 = int(state['track_bbs_ids'][-1][1])
                x2 = int(state['track_bbs_ids'][-1][2])
                y2 = int(state['track_bbs_ids'][-1][3])
                cv2.rectangle(im, (x1, y1), (x2, y2), (255,0,0), 2)
                
            cv2.imshow('SiamMask', im)
            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
