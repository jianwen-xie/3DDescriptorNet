from __future__ import division
from __future__ import print_function

import os
import argparse
import math
import numpy as np
import tensorflow as tf

from util import data_io
from models.desnet3d import DescriptorNet3D
from progressbar import ETA, Bar, Percentage, ProgressBar

parser = argparse.ArgumentParser()

# CoopNet hyper-parameters
parser.add_argument('-sz', '--cube_len', type=int, default=32)

# training hyper-parameters
parser.add_argument('--num_epochs', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--sample_batch', type=int, default=25)
parser.add_argument('--d_lr', type=float, default=0.01)
parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

# langevin hyper-parameters
parser.add_argument('--refsig', type=float, default=0.5)  # 0.005 0.0017
parser.add_argument('-delta', '--step_size', type=float, default=0.1)  # 0.005 0.0017
parser.add_argument('--sample_steps', type=int, default=20)

# util
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--category', type=str, default='bathtub')
parser.add_argument('--data_path', type=str,
                    default='/home/zilong/Documents/3DCoopNet/volumetric_data/ModelNet10')
parser.add_argument('--log_step', type=int, default=10)

def main(_):
    RANDOM_SEED = 66
    np.random.seed(RANDOM_SEED)

    opt = parser.parse_args()

    output_dir = os.path.join(opt.output_dir, opt.category)
    sample_dir = os.path.join(output_dir, 'synthesis')
    log_dir = os.path.join(output_dir, 'log')
    model_dir = os.path.join(output_dir, 'checkpoints')

    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    if tf.gfile.Exists(sample_dir):
        tf.gfile.DeleteRecursively(sample_dir)
    tf.gfile.MakeDirs(sample_dir)

    if tf.gfile.Exists(model_dir):
        tf.gfile.DeleteRecursively(model_dir)
    tf.gfile.MakeDirs(model_dir)

    # Prepare training data
    train_data = data_io.getObj(opt.data_path, opt.category, cube_len=opt.cube_len,
                                low_bound=0, up_bound=1)

    data_io.saveVoxelsToMat(train_data, "%s/observed_data.mat" % output_dir, cmin=0, cmax=1)

    # Preprocess training data
    voxel_mean = train_data.mean()
    train_data = train_data - voxel_mean
    train_data = train_data[..., np.newaxis]

    opt.num_batches = int(math.ceil(len(train_data) / opt.batch_size))

    print('Reading voxel data {}, shape: {}'.format(opt.category, train_data.shape))
    print('min: %.4f\tmax: %.4f\tmean: %.4f' % (train_data.min(), train_data.max(), voxel_mean))

    # create and build model
    net = DescriptorNet3D(opt)
    net.build_model()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sample_size = opt.sample_batch * opt.num_batches
        sample_voxels = np.random.randn(sample_size, opt.cube_len, opt.cube_len, opt.cube_len, 1)

        saver = tf.train.Saver(max_to_keep=50)

        writer = tf.summary.FileWriter(log_dir, sess.graph)

        for epoch in range(opt.num_epochs):
            d_grad_acc = []

            widgets = ["Epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval=opt.num_batches, widgets=widgets)
            pbar.start()

            sess.run(net.reset_grads)
            for i in range(opt.num_batches):
                obs_data = train_data[i * opt.batch_size:min(len(train_data), (i + 1) * opt.batch_size)]
                syn_data = sample_voxels[i * opt.sample_batch:(i + 1) * opt.sample_batch]

                # generate synthesized images
                if epoch <= 100:
                    syn = sess.run(net.langevin_descriptor_noise, feed_dict={net.syn: syn_data})

                else:
                    syn = sess.run(net.langevin_descriptor, feed_dict={net.syn: syn_data})

                # learn D net
                d_grad = sess.run([net.des_grads, net.des_loss_update, net.update_d_grads, net.sample_loss_update],
                                       feed_dict={net.obs: obs_data, net.syn: syn})[0]

                d_grad_acc.append(d_grad)

                # Compute L2 distance
                sess.run(net.recon_err_update, feed_dict={net.obs: obs_data, net.syn: syn})

                sample_voxels[i * opt.sample_batch:(i + 1) * opt.sample_batch] = syn

            pbar.finish()

            sess.run(net.apply_d_grads)
            [des_loss_avg, sample_loss_avg, mse, summary] = sess.run([net.des_loss_mean, net.sample_loss_mean,
                                                                      net.recon_err_mean, net.summary_op])

            print('Epoch #%d, descriptor loss: %.4f, descriptor SSD weight: %.4f, sample loss: %.4f, Avg MSE: %4.4f'
                  % (epoch, des_loss_avg, float(np.mean(d_grad_acc)), sample_loss_avg, mse))
            writer.add_summary(summary, epoch)

            if epoch % opt.log_step == 0:
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)
                data_io.saveVoxelsToMat(sample_voxels + voxel_mean, "%s/sample%04d.mat" % (sample_dir, epoch),
                                        cmin=0, cmax=1)

                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                saver.save(sess, "%s/%s" % (model_dir, 'net.ckpt'), global_step=epoch)


if __name__ == '__main__':
    tf.app.run(main)
