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


## Contents
1. [Environment Setup](#environment-setup)
2. [Demo](#demo)
3. [Testing Models](#testing-models)
4. [Training Models](#training-models)

## Environment setup
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1, CUDA 11.4, RTX 2060 GPUs

- Clone the repository 
```
git clone https://github.com/mhd-medfa/SiamMask.git && cd SiamMask
export SiamMask=$PWD
```
- Setup python environment
```
conda create -n siammask python=3.6 anaconda
source activate siammask
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip3 install opencv-python
pip3 install cython
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
cd $SiamMask/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```

- 
- Run `run.py`

```shell
cd $SiamMask/experiments/siammask_sharp
export PATH="/root/anaconda3/bin:$PATH"
export PYTHONPATH="/root/anaconda3/envs/siammask/bin/python3.6"
source activate siammask
python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_davis.json
```
Or

- Run `realtime_demo.py`

```shell
cd $SiamMask/experiments/siammask_sharp
export PATH="/root/anaconda3/bin:$PATH"
export PYTHONPATH="/root/anaconda3/envs/siammask/bin/python3.6"
source activate siammask
python ../../tools/realtime_demo.py --resume SiamMask_DAVIS.pth --config config_davis.json
```

<div align="center">
  <img src="http://www.robots.ox.ac.uk/~qwang/SiamMask/img/SiamMask_demo.gif" width="500px" />
</div>


## Testing
- [Setup](#environment-setup) your environment
- Download test data
```shell
cd $SiamMask/data
sudo apt-get install jq
bash get_test_data.sh
```
- Download pretrained models
```shell
cd $SiamMask/experiments/siammask_sharp
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


## License
Licensed under an MIT license.

