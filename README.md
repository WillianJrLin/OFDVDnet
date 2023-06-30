# OFDVDnet
___
A denoising algorithm for Poisson noise corrupted videos utilizing optical flow motion compensation and sensor fusion of a CMOS camera and a shot noise limited camera (such as SPAD camera).


## Overview
___
This repository contains the original code implementation for the paper ["OFDVDnet: A Sensor Fusion Approach for Video Denoising in Fluorescence-Guided Surgery" Medical Imaging with Deep Learning (2023)](https://openreview.net/forum?id=TcUtCXRcK8), as well as our in-house captured flourescence guilded surgery (FGS) dataset for training and evaluations. 

OFDVDnet consists of two stages:
  - Optical flow denoising 
  - Neural network denoising

An overview of OFDVDnet's denoising pipeline is shown in the figure below.


> <img src="./figures/teaser.png" width="700">


## Environment Requirement
___
  - Python 3
  - Pytorch
  - Numpy
  - Matplotlib
  - OpenCV 


## Dataset
___


## Usage
___
### Step 1: Optical Flow Denoising
### Step 2: Neural Network Denoising


## Results
___
| SBR | OFDVDnet      | OFDV          | FastDVDnet    | V-BM4D        | Guided Filtering| Joint Bilateral |
|:---:|     :---:     |     :---:     |     :---:     |     :---:     |      :---:      |      :---:      |
| 0.1 | 29.3/.76/.88  | 10.8/.015/.20 | 24.3/.48/.83  | 19.7/.19/.52  | 16.4/.19/.69    | 15.8/.11/.59    |
| 0.5 | 34.0/.89/.93  | 21.5/.22/.52  | 30.8/.80/.88  | 29.9/.61/.86  | 28.1/.61/.86    | 26.3/.52/.81    |
| 2.0 | 36.9/.92/.95  | 30.8/.72/.82  | 35.7/.89/.93  | 36.7/.88/.92  | 33.7/.90/.92    | 31.5/.85/.90    |


![vid323_01](https://www.youtube.com/watch?v=m0wEeO6K4-E&list=PLyMBGD47PvgjNq__m8VCVDN1O9L8Jnpg2&index=3)



> <img src="./figures/results.png" width="500">

### Limitations
> <img src="./figures/limits.png" width="500">

## License 
___ 