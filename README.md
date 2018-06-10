# Learning Descriptor Networks for 3D Shape Synthesis and Analysis

This repository contains a tensorflow implementation for the paper "[Learning Descriptor Networks for 3D Shape Synthesis and Analysis](http://www.stat.ucla.edu/~jxie/3DDescriptorNet/3DDescriptorNet_file/doc/3DDescriptorNet.pdf)
". (http://www.stat.ucla.edu/~jxie/3DDescriptorNet/3DDescriptorNet.html)

<p align="center"><img src="http://www.stat.ucla.edu/~jxie/3DDescriptorNet/files/syn.jpg" width="700px"/></p>

## Requirements
- Python 2.7 or Python 3.3+
- [Tensorflow r1.3+](https://www.tensorflow.org/install/)
- Install required python libraries
```bash
pip install numpy scipy
```

## Usage
- Clone this repo:
```bash
git clone https://github.com/jianwen-xie/3DDescriptorNet.git
cd 3DDescriptorNet
```

- Download [volumetric data](https://drive.google.com/file/d/1fwYcL9KMWW1aX3r6hPCGC7VYpF5BzHjS/view?usp=sharing) and save it to `./data` directory:

- Train the synthesis model with ***night_stand*** dataset:
```bash
python train.py --category night_stand --data_dir ./data/volumetric_data/ModelNet10 --output_dir ./output
```

- Visualize the generated results in using the visualization code in `visualization/visualize.m`, e.g.
```MATLAB
addpath('visualization')
visualize('./output/night_stand/synthesis', 'sample2990.mat')
```

## References
    @inproceedings{3DDesNet,
        title={Learning Descriptor Networks for 3D Shape Synthesis and Analysis},
        author={Xie, Jianwen and Zheng, Zilong and Gao, Ruiqi and Wang, Wenguan and Zhu Song-Chun and Wu, Ying Nian},
        booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2018}
    }
For any questions, please contact Jianwen Xie (jianwen@ucla.edu) and Zilong Zheng (zilongzheng0318@ucla.edu).
