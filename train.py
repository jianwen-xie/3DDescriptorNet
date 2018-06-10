from __future__ import division
from __future__ import print_function

import os
import argparse
import math
import time
import numpy as np
import tensorflow as tf
from config import FLAGS

from util import data_io
from models.desnet3d import DescriptorNet3D

tf.flags.DEFINE_integer('sample_batch', 25, 'Number of samples synthesized for each batch')

def main(_):
    RANDOM_SEED = 66
    np.random.seed(RANDOM_SEED)

    output_dir = os.path.join(FLAGS.output_dir, FLAGS.category)
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
    train_data = data_io.getObj(FLAGS.data_path, FLAGS.category, cube_len=FLAGS.cube_len, num_voxels=FLAGS.train_size,
                                low_bound=0, up_bound=1)

    data_io.saveVoxelsToMat(train_data, "%s/observed_data.mat" % output_dir, cmin=0, cmax=1)

    # Preprocess training data
    voxel_mean = train_data.mean()
    train_data = train_data - voxel_mean
    train_data = train_data[..., np.newaxis]

    FLAGS.num_batches = int(math.ceil(len(train_data) / FLAGS.batch_size))

    print('Reading voxel data {}, shape: {}'.format(FLAGS.category, train_data.shape))
    print('min: %.4f\tmax: %.4f\tmean: %.4f' % (train_data.min(), train_data.max(), voxel_mean))

    # create and build model
    net = DescriptorNet3D(FLAGS)
    net.build_model()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sample_size = FLAGS.sample_batch * FLAGS.num_batches
        sample_voxels = np.random.randn(sample_size, FLAGS.cube_len, FLAGS.cube_len, FLAGS.cube_len, 1)

        saver = tf.train.Saver(max_to_keep=50)

        writer = tf.summary.FileWriter(log_dir, sess.graph)

        for epoch in range(FLAGS.num_epochs):
            d_grad_acc = []

            start_time = time.time()

            sess.run(net.reset_grads)
            for i in range(FLAGS.num_batches):

                obs_data = train_data[i * FLAGS.batch_size:min(len(train_data), (i + 1) * FLAGS.batch_size)]
                syn_data = sample_voxels[i * FLAGS.sample_batch:(i + 1) * FLAGS.sample_batch]

                # generate synthesized images
                if epoch < 100:
                    syn = sess.run(net.langevin_descriptor_noise, feed_dict={net.syn: syn_data})

                else:
                    syn = sess.run(net.langevin_descriptor, feed_dict={net.syn: syn_data})

                # learn D net
                d_grad = sess.run([net.des_grads, net.des_loss_update, net.update_d_grads, net.sample_loss_update],
                                       feed_dict={net.obs: obs_data, net.syn: syn})[0]

                d_grad_acc.append(d_grad)

                # Compute L2 distance
                sess.run(net.recon_err_update, feed_dict={net.obs: obs_data, net.syn: syn})

                sample_voxels[i * FLAGS.sample_batch:(i + 1) * FLAGS.sample_batch] = syn

            sess.run(net.apply_d_grads)
            [des_loss_avg, sample_loss_avg, mse, summary] = sess.run([net.des_loss_mean, net.sample_loss_mean,
                                                                      net.recon_err_mean, net.summary_op])
            end_time = time.time()

            print('Epoch #%d, descriptor loss: %.4f, descriptor SSD weight: %.4f, sample loss: %.4f, Avg MSE: %4.4f, time: %.2fs'
                  % (epoch, des_loss_avg, float(np.mean(d_grad_acc)), sample_loss_avg, mse, end_time - start_time))
            writer.add_summary(summary, epoch)

            if epoch % FLAGS.log_step == 0:
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)
                data_io.saveVoxelsToMat(sample_voxels + voxel_mean, "%s/sample%04d.mat" % (sample_dir, epoch),
                                        cmin=0, cmax=1)

                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                saver.save(sess, "%s/%s" % (model_dir, 'net.ckpt'), global_step=epoch)


if __name__ == '__main__':
    tf.app.run(main)
