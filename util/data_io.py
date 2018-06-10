from __future__ import division
from __future__ import print_function

import os

import scipy.io as io
import numpy as np
import scipy.ndimage as nd

DATA_PATH = './volumetric_data/ModelNet10/'

def clip_by_value(input_, clip_value_min=0, clip_value_max=1):
    return np.maximum(clip_value_min, np.minimum(clip_value_max, input_))

def saveVoxelsToMat(voxels, path, data_name='voxels', cmin=0, cmax=1):
    saved_voxel = np.zeros(voxels.shape)
    if len(voxels.shape) == 3:
        saved_voxel = normalize_data(clip_by_value(voxels, cmin, cmax), 0, 1)
    else:
        for i in range(len(voxels)):
            saved_voxel[i] = normalize_data(clip_by_value(voxels[i], cmin, cmax), 0, 1)
    io.savemat(path, {data_name: saved_voxel})

def normalize_data(data, low_bound=0, up_bound=1):
    min_val = data.min()
    max_val = data.max()
    if max_val - min_val == 0:
        return 0
    return low_bound + (data - min_val) / (max_val - min_val) * (up_bound - low_bound)

def get_incomplete_data(voxels, p=0.3):
    num_points = int(np.prod(voxels.shape))
    num_zeros = int(num_points * p)
    mask = np.append(np.zeros(num_zeros), np.ones(num_points - num_zeros))
    np.random.shuffle(mask)
    mask = mask.reshape(*voxels.shape)
    return voxels * (1 - mask), mask

def getVoxelsFromMat(path, data_name='instance', low_bound=0, up_bound=1):
    voxels = np.array(io.loadmat(path)[data_name], dtype=np.float32)
    if len(voxels.shape) == 3:
        voxels = normalize_data(voxels, low_bound, up_bound)
    else:
        for i in range(len(voxels)):
            voxels[i] = normalize_data(voxels[i], low_bound, up_bound)
    return voxels


def getObj(data_path, obj='chair', train=True, num_voxels=None, cube_len=64, low_bound=0, up_bound=1):
    objPath = os.path.join(data_path, obj, str(cube_len))
    objPath += '/train/' if train else '/test/'
    fileList = [f for f in os.listdir(objPath) if f.endswith('.mat')]
    fileList.sort()
    volumeBatch = np.asarray([getVoxelsFromMat(objPath + f, low_bound=low_bound, up_bound=up_bound) for f in fileList],
                             dtype=np.float32)
    if num_voxels != None:
        volumeBatch = volumeBatch[:num_voxels, :, :, :]
    print('Loading {}, shape: {}'.format(obj, volumeBatch.shape))
    return volumeBatch


def getAll(data_path, train=True, cube_len=64):
    print('Loading all data sets......')
    objList = [obj for obj in os.listdir(data_path)]
    volumeBatch = np.concatenate([getObj(data_path, obj, train, cube_len=cube_len) for obj in objList], axis=0)
    np.random.shuffle(volumeBatch)
    print('All data sets loaded, shape: ', volumeBatch.shape)
    return volumeBatch.astype(float)

def reshape_data(data_dir, filename, rand_voxels=None):
    voxels = getVoxelsFromMat(os.path.join(data_dir, filename), 'voxels')
    voxels = np.array(voxels > 0.5).astype(float)
    scale = 24 / 32
    if  rand_voxels != None:
        idx = np.random.randint(voxels.shape[0], size=rand_voxels)
        voxels = voxels[idx]
        saveVoxelsToMat(voxels, os.path.join(data_dir, 'samples.mat'), 'voxels')
    num_voxels = voxels.shape[0]
    new_voxel = np.zeros(shape=(num_voxels, 30, 30, 30))
    for i in range(num_voxels):
        voxel = np.squeeze(voxels[i])
        voxel = nd.zoom(voxel, (scale, scale, scale), mode='nearest', order=3)
        voxel = np.pad(voxel, 3, mode='constant', constant_values=0)
        new_voxel[i] = voxel
    new_voxel = np.array(new_voxel > 0.5).astype(float)
    saveVoxelsToMat(new_voxel, os.path.join(data_dir, 'syn_results.mat'), 'v')

