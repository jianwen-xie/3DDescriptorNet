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

- Download [volumetric data](https://drive.google.com/file/d/0B9FKAOJSlMq0Vnl2WlN3eU40RGs/view?usp=sharing) and save it to `./data` directory. 
The dataset contains 10 categories of voxelizations of [ModelNet10](http://3dshapenets.cs.princeton.edu/ModelNet10.zip).

### Exp1: 3D Object synthesis

- Train the synthesis model on **night stand** category:
    ```bash
    python train.py --category night_stand --data_dir ./data/volumetric_data/ModelNet10 --output_dir ./output
    ```

- Visualize the generated results using the *MATLAB* code in `visualization/visualize.m`, e.g.
    ```MATLAB
    addpath('visualization')
    visualize('./output/night_stand/synthesis', 'sample2990.mat')
    ```

- Evaluate synthesized results using the evaluation code in `./evaluation`

- You can download our [synthesized results](https://drive.google.com/file/d/1o1Q_DEE4YPVVl89_vPYVJrPTmOh5gZdq/view?usp=sharing) and test on it.

### Exp2: 3D object recovery
<p align="center"><img src="http://www.stat.ucla.edu/~jxie/3DDescriptorNet/files/sofa.jpg" width="400px"/></p>

- Train the recovery model on **sofa** category:
    ```bash
    python rec_exp.py --category sofa \
                      --num_epochs 1000 \
                      --batch_size 50 \
                      --step_size 0.07 \
                      --sample_steps 90 
    ```

- Test the recovery model:
    1. Download the [incomplete data](https://drive.google.com/file/d/1Q-tapylbCcS-i7IWPKNaPG9c4hLfi7I_/view?usp=sharing) and save it to `./data` directory. For each category in `volumetric_data`, the 
    incomplete data contains: 1) `incomplete_test.mat`: 70\% randomly corrupted testing data 2) `masks.mat`: The mask to corrupt the testing data. 3. `original_test.mat`: original testing data for comparison.
    2. You can download our [pretrained model](https://drive.google.com/file/d/1cm8Q8JaLBf8h76g1bfnjWBl6tZmbZOuL/view?usp=sharing) to test recovery.
    2. Run recovery on the corrupted data
    ```bash
    python rec_exp.py --test --category sofa \
                      --ckpt ./recovery_model/sofa/sofa.ckpt \
                      --incomp_data_path ./data/incomplete_data \
                      --batch_size 50 \
                      --step_size 0.07 \
                      --sample_steps 90 
    ```

### Exp3: 3D object super resolution
<p align="center"><img src="http://www.stat.ucla.edu/~jxie/3DDescriptorNet/files/3D_sr.png" width="400px"/></p>

- Train the super resolution model on **toilet** category:
    ```bash
    python sr_exp.py --category toilet \
                      --cube_len 64 \
                      --scale 4 \
                      --num_epochs 500 \
                      --batch_size 50 \
                      --step_size 0.01 \
                      --sample_steps 10 
    ```

- Test the super resolution model:
    ```bash
    python rec_exp.py --test --category toilet \
                      --ckpt ./recovery_model/sofa/sofa.ckpt \
                      --cube_len 64 \
                      --scale 4 \
                      --batch_size 50 \
                      --step_size 0.01 \
                      --sample_steps 10 
    ```


## References
    @inproceedings{3DDesNet,
        title={Learning Descriptor Networks for 3D Shape Synthesis and Analysis},
        author={Xie, Jianwen and Zheng, Zilong and Gao, Ruiqi and Wang, Wenguan and Zhu Song-Chun and Wu, Ying Nian},
        booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2018}
    }
For any questions, please contact Jianwen Xie (jianwen@ucla.edu) and Zilong Zheng (zilongzheng0318@ucla.edu).
