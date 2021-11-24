# DANCER-FLOW MODULE

## MMCV, MMDET, MMPOSE
## Requirements
- window10, mac
- python 3.8
- git
- pytorch 1.9
- CUDA 11.4
- opencv-python
- [xtcocotools](https://github.com/jin-s13/xtcocoapi)
- [mmcv](https://github.com/open-mmlab/mmcv.git)
- [mmdet](https://github.com/open-mmlab/mmdetection)
- [mmpose](https://github.com/open-mmlab/mmpose.git)

## Quick install
a. Create a conda virtual enviroment and activate it.

```shell
conda create -n dancer-flow python=3.8 -y
conda activate dacer-flow
```

b. Install [Pytorch](https://pytorch.org/)
```shell
conda install pytorch torchvision torchaudio -c pytorch
```

c. Install Opencv-python
```shell
pip install opencv-python
```

d. Install xtcocotools
```shell
git clone https://github.com/jin-s13/xtcocoapi.git
cd xtcocotools
pip install -r requirements.txt
pip install -v -e .
cd ..
```

e. Install MMCV
```shell
pip install mmcv-full
```

f. Install mmdet
```shell
pip install mmdet
```

g. Install mmpose
```shell
pip install mmpose
```