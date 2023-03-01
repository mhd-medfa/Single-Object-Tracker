# 3D-SiamMask


This is the official implementation for 3D-SiamMask (RemoteSensing2022). For technical details, please refer to:

**3D-SiamMask: Vision-Based Multi-Rotor Aerial-Vehicle Tracking for a Moving Object** <br />
[Mohamad Al Mdfaa](https://mhd-medfa.github.io/)\*, [Geesara Kulathunga](https://scholar.google.ru/citations?user=6VtrN-MAAAAJ&hl=en)\*, [Alexandr Klimchik](https://scholar.google.fr/citations?user=KLpMBj0AAAAJ&hl=en)\* (\* denotes equal contribution) <br />
**Remote Sensing 2022** <br />
**[[Paper](https://www.mdpi.com/1945298)] [[Video](https://youtu.be/za2jyAssKWE)] [[Project Page](https://sites.google.com/view/3d-siammask/home)]** <br />


### Bibtex
If you find this code useful, please consider citing:

```
@article{al20223d,
  title={3D-SiamMask: Vision-Based Multi-Rotor Aerial-Vehicle Tracking for a Moving Object},
  author={Al Mdfaa, Mohamad and Kulathunga, Geesara and Klimchik, Alexandr},
  journal={Remote Sensing},
  volume={14},
  number={22},
  pages={5756},
  year={2022},
  publisher={MDPI}
}

```

[![Watch This Video](https://img.youtube.com/vi/za2jyAssKWE/0.jpg)](https://youtu.be/za2jyAssKWE)

## Contents
1. [Environment Setup](#environment-setup)
2. [Demo](#demo)
3. [Testing Models](#testing-models)
4. [Training Models](#training-models)

## Environment setup
This code has been tested on Docker, Ubuntu 18.04, Python 3.6, Pytorch 0.4.1, CUDA 11.4, RTX 2060 GPUs

### **Method 1 - Recommended**
- Pull the docker image

```shell
docker pull medfa1/3d-siammask:latest
```

```shell
docker run -itd --name sot  --privileged \
    --net=host --gpus all \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \ medfa1/3d-siammask:latest
```

### **Method 2**
- Clone the TrajectoryTracker repository

```shell
git clone https://github.com/GPrathap/trajectory-tracker.git
```
, and follow instructions in README.md to build the environment.

- Clone the CustomRobots repository and utilize `car_junctuion/gas_station`

```shell
git clone https://github.com/JdeRobot/CustomRobots.git
```
This step assumes that the have experience with ROS and you know what to do.

- Clone the repository 
```shell
git clone https://github.com/mhd-medfa/Single-Object-Tracker.git && cd Single-Object-Tracker
export SOT=$PWD
```
- Setup python environment
```
conda create -n siammask python=3.6 anaconda
source activate siammask
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip3 install opencv-python
pip3 install cython
pip3 install pykalman
bash make.sh
```
- Add the project to your PYTHONPATH
```
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Demo
- [Setup](#environment-setup) your environment
- Download the SiamMask model
```shell
cd $SOT/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
- Check the local_planner
```shell
xhost +
docker exec -it sot bash
```

Now to run the local planner you need to run the following command as explained in the [video](https://www.youtube.com/watch?v=12MtXTtKRBE):
```
1- roslaunch drone_sim sim.launch
2- roslaunch state_machine take_off.launch
3- roslaunch state_machine rviz_p4.launch
4- roslaunch state_machine fsm_trajectory_point_stabilizer.launch
5- roslaunch state_machine px4_reg.launch
```
[Watch the video on YouTube](https://www.youtube.com/watch?v=12MtXTtKRBE)

- Run `run.py`

If the local planner in the previous step works well, re-run only the instructions `(1, 2, 3, and 4)` also run the Single-Object Tracker

```shell
cd $SOT/tools
export PATH="/root/anaconda3/bin:$PATH"
export PYTHONPATH="/root/anaconda3/envs/siammask/bin/python3.6"
source activate siammask
python run.py
```
After selecting the object, run `5` and don't forget to press Publish Waypoints in rviz and to stop the take_off.launch `(3)`.

## Testing
- [Setup](#environment-setup) your environment
- Download test data
```shell
cd $SOT/data
sudo apt-get install jq
bash get_test_data.sh
```
- Download pretrained models
```shell
cd $SOT/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT_LD.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
- Evaluate performance on [VOT](http://www.votchallenge.net/)
```shell
bash test_mask_refine.sh config_vot.json SiamMask_VOT.pth VOT2016 0
bash test_mask_refine.sh config_vot.json SiamMask_VOT.pth VOT2018 0
bash test_mask_refine.sh config_vot.json SiamMask_VOT.pth VOT2019 0
bash test_mask_refine.sh config_vot18.json SiamMask_VOT_LD.pth VOT2016 0
bash test_mask_refine.sh config_vot18.json SiamMask_VOT_LD.pth VOT2018 0
python ../../tools/eval.py --dataset VOT2016 --tracker_prefix C --result_dir ./test/VOT2016
python ../../tools/eval.py --dataset VOT2018 --tracker_prefix C --result_dir ./test/VOT2018
python ../../tools/eval.py --dataset VOT2019 --tracker_prefix C --result_dir ./test/VOT2019
```
- Evaluate performance on [DAVIS](https://davischallenge.org/) (less than 50s)
```shell
bash test_mask_refine.sh config_davis.json SiamMask_DAVIS.pth DAVIS2016 0
bash test_mask_refine.sh config_davis.json SiamMask_DAVIS.pth DAVIS2017 0
```
- Evaluate performance on [Youtube-VOS](https://youtube-vos.org/) (need download data from [website](https://youtube-vos.org/dataset/download))
```shell
bash test_mask_refine.sh config_davis.json SiamMask_DAVIS.pth ytb_vos 0
```

## References
The local planner is based on this repo:
https://github.com/GPrathap/trajectory-tracker

The tracker was mostly inspired by SiamMask work:
https://github.com/foolwood/SiamMask

For the full list of  references, check out the paper.
## License
Licensed under an MIT license.

