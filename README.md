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

Coming soon.

[//]: # (# Installation)

[//]: # ()
[//]: # (## Environment)

[//]: # (* Tested OS: Ubuntu 18.04 LTS / RTX3090)

[//]: # (* Python >= 3.7)

[//]: # (* PyTorch == 1.8.0)

[//]: # ()
[//]: # (## Dependencies)

[//]: # (1. Install [PyTorch 1.8.0]&#40;https://pytorch.org/get-started/previous-versions/&#41; with the correct CUDA version.)

[//]: # (2. Install the dependencies:)

[//]: # (    ```)

[//]: # (    pip install -r requirements.txt)

[//]: # (    ```)

[//]: # ()
[//]: # (# Evaluation)

[//]: # (Download the pre-trained models from [GoogleDrive]&#40;https://drive.google.com/file/d/11zNG_QMD8oXQwx46S6FY2z5Hqsnmh7rD/view?usp=sharing&#41;. Then unzip and put it under the project folder.)

[//]: # (Run the following and then you will be able to reproduce the main results in our paper. )

[//]: # (<dataset_name> can be eth, hotel, univ, zara1, zara2 and sdd.)

[//]: # (```)

[//]: # (python test.py --dataset <dataset_name> --gpu <gpu_id>)

[//]: # (```)

[//]: # ()
[//]: # (# Training)

[//]: # (This model requires **two-stage** training.)

[//]: # (1. Train the Multi-stream Representation Learning based CVAE model)

[//]: # (    ```)

[//]: # (    python trainvae.py --dataset <dataset_name> --gpu <gpu_id>)

[//]: # (    ```)

[//]: # (2. Train the sampler model)

[//]: # (    ```)

[//]: # (    python trainsampler.py --dataset <dataset_name> --gpu <gpu_id>)

[//]: # (    ```)

[//]: # (You can modify the configuration by giving different parameters.)

[//]: # ()
[//]: # (# Acknowledgement)

[//]: # (Thanks for the ETH-UCY data processing from [SGCN]&#40;https://github.com/shuaishiliu/SGCN&#41; and SDD data provided by [PECNet]&#40;https://github.com/j2k0618/PECNet_nuScenes&#41;.)
