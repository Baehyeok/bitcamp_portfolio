# Dance-Performance-Score

<br>

# 1. Get_Start
## Model_Server Run
```shell
cd model_server
python model.py
```
## Web_Server Run
```shell
cd Dancer-Flow
cd app
python manage.py migrate
python manage.py runserver
```
<br>

# 2. Prerequisites
- Window or macOS
- python 3.8+
- PyTorch 1.9.0+
- CUDA .V
- MMCV
- MMDetection
- MMPose
- Django
- DataBase :

<br>

# 3. Installation
### Prepare environment
1. Create a conda virtual environment and activate it.
```shell
conda create -n dancerflow python=3.8 -y
conda activate dancerflow
```

2. Install PyTorch and torchvision following the official instructions
```shell
conda install pytorch torchvison torchaudio -c pytorch
```