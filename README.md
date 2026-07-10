# Info
This is the code and model weights of our paper: MSO-Det: A Multi-Level Synergistic Optimization Framework for Robust Spike Object Detection in Field Scenes.

## How to use
### Setup environment
Create a conda environment which using python 3.11, and install libraries.

Install `PyTorch 2.2.1`, you can choose CPU or GPU version according to your device.

then
```plain
pip install -r requirements.txt
```
### Data
Place your data in `data` floder and set the path in `configs\dataset\custom_detection.yml` file.

### train
One GPU:
```
python train.py -c configs/yaml/mso-det.yml
```
Multi GPUs:
```
torchrun --master_port=9940 --nproc_per_node=$NUM_GPUS train.py -c configs/yaml/mso-det.yml
```

### test
One GPU:
```
python train.py -c configs/yaml/mso-det.yml -r weights/mso-det.pth --test-only 
```
Multi GPUs:
```
torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/yaml/mso-det.yml -r weights/mso-det.pth --test-only
```

You can find the results in `outputs` folder in default.