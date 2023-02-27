# Multi-stream Representation Learning for Pedestrian Trajectory Prediction
Official code for AAAI 2023 paper "Multi-stream Representation Learning for Pedestrian Trajectory Prediction"

# Framework

## Multi-stream Representation Learning
<div align='center'>
<img src="figures/MSRL.jpg"></img>
</div>
<br />

## CVAE for Multi-modal Prediction
<div align='center'>
<img src="figures/CVAE.jpg"></img>
</div>
<br />

# Installation

## Environment
* Tested OS: Ubuntu 18.04 LTS / RTX3090
* Python >= 3.7
* PyTorch == 1.8.0

## Dependencies
1. Install [PyTorch 1.8.0](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

# Evaluation
Download the pre-trained models from BaiduYun. Then unZip and put it in the project folder.
Run the following and you will be able to reproduce the main result in our paper.
```
python test.py --dataset <dataset_name> --gpu <gpu_id>
```

# Training

# Acknowledge
