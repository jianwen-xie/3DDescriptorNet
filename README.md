# Learning Descriptor Networks for 3D Shape Synthesis and Analysis

This repository contains a tensorflow implementation for the paper "[Learning Descriptor Networks for 3D Shape Synthesis and Analysis](http://www.stat.ucla.edu/~jxie/3DDescriptorNet/3DDescriptorNet_file/doc/3DDescriptorNet.pdf)
". (http://www.stat.ucla.edu/~jxie/3DDescriptorNet/3DDescriptorNet.html)

<p align="center"><img src="http://www.stat.ucla.edu/~jxie/3DDescriptorNet/files/syn.jpg" width="700px"/></p>

## Requirements
- Python 2.7 or Python 3.3+
- [Tensorflow r1.3+](https://www.tensorflow.org/install/)
- Install required Python libraries
    ```bash
    pip install numpy scipy
    ```

## Getting Started

- Clone this repo:
    ```bash
    git clone https://github.com/jianwen-xie/3DDescriptorNet.git
    cd 3DDescriptorNet
    ```

- Download [volumetric data](https://drive.google.com/file/d/1fwYcL9KMWW1aX3r6hPCGC7VYpF5BzHjS/view?usp=sharing) and save it to `./data` directory. 
The dataset contains 10 categories of voxelizations of [ModelNet10](http://3dshapenets.cs.princeton.edu/ModelNet10.zip). We use ***dresser*** as an example for the following experiments.

### Exp1: 3D Object synthesis

- Train the synthesis model:
    ```bash
    python train.py --category dresser --data_dir ./data/volumetric_data/ModelNet10 --output_dir ./output
    ```

- Visualize the generated results using the visualization code in `visualization/visualize.m`, e.g.
    ```MATLAB
    addpath('visualization')
    visualize('./output/dresser/synthesis', 'sample2990.mat')
    ```

- Evaluate synthesized results using the evaluation code in `./evaluation`

### Exp2: 3D object recovery

- Train the recovery model:
    ```bash
    python rec_exp.py --category dresser \
                      --num_epochs 1000 \
                      --batch_size 50 \
                      --step_size 0.07 \
                      --sample_steps 90 
    ```

- Test the recovery model:
    1. Download the [incomplete data](https://drive.google.com/file/d/1Q-tapylbCcS-i7IWPKNaPG9c4hLfi7I_/view?usp=sharing) and save it to `./data` directory. For each category in `volumetric_data`, the 
    incomplete data contains: 1) `incomplete_test.mat`: 70\% randomly corrupted testing data 2) `masks.mat`: The mask to corrupt the testing data. 3. `original_test.mat`: original testing data for comparison.
    2. Run recovery on the corrupted data
    ```bash
    python rec_exp.py --test \
                      --ckpt ./output/dresser/checkpoint/model.ckpt-990 \
                      --incomp_data_path ./data/incomplete_data \
                      --category dresser \
                      --batch_size 50 \
                      --step_size 0.07 \
                      --sample_steps 90 
    ```


## References
    @inproceedings{3DDesNet,
        title={Learning Descriptor Networks for 3D Shape Synthesis and Analysis},
        author={Xie, Jianwen and Zheng, Zilong and Gao, Ruiqi and Wang, Wenguan and Zhu Song-Chun and Wu, Ying Nian},
        booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2018}
    }
For any questions, please contact Jianwen Xie (jianwen@ucla.edu) and Zilong Zheng (zilongzheng0318@ucla.edu).
