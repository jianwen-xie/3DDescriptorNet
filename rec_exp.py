from __future__ import division
from __future__ import print_function

import math
import os
import time
import matplotlib.pyplot as plt
from scipy import io
from config import FLAGS
from util import data_io
from util.custom_ops import *

tf.flags.DEFINE_string('incomp_data_path', './data/incomplete_data', 'Path to incomplete data')


def get_incomplete_data(voxels, p=0.3):
    num_points = int(np.prod(voxels.shape))
    num_ones = int(num_points * p)
    mask = np.append(np.ones(num_ones), np.zeros(num_points - num_ones))
    np.random.shuffle(mask)
    mask = mask.reshape(*voxels.shape)
    voxel_mean = np.sum(voxels * mask) / np.sum(mask)
    return voxels * mask + np.ones(shape=voxels.shape) * voxel_mean * (1 - mask), mask


def descriptor(inputs, reuse=False):
    with tf.variable_scope('des', reuse=reuse):
        conv1 = conv3d(inputs, 200, kernal=(16, 16, 16), strides=(3, 3, 3), padding="SAME", name="conv1")
        conv1 = tf.nn.relu(conv1)

        conv2 = conv3d(conv1, 100, kernal=(6, 6, 6), strides=(2, 2, 2), padding="SAME", name="conv2")
        conv2 = tf.nn.relu(conv2)

        fc = fully_connected(conv2, 1, name="fc")

        return fc


def langevin_dynamics(syn_arg):
    def cond(i, syn):
        return tf.less(i, FLAGS.sample_steps)

    def body(i, syn):
        syn_res = descriptor(syn, reuse=True)
        grad = tf.gradients(syn_res, syn, name='grad_des')[0]
        syn = syn - 0.5 * FLAGS.step_size * FLAGS.step_size * (syn / FLAGS.refsig / FLAGS.refsig - grad)
        return tf.add(i, 1), syn

    with tf.name_scope("langevin_dynamics"):
        i = tf.constant(0)
        i, syn = tf.while_loop(cond, body, [i, syn_arg])
        return syn


def train():
    cube_len = FLAGS.cube_len
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.category)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    synthesis_dir = os.path.join(output_dir, 'recovery')
    log_dir = os.path.join(output_dir, 'log')

    obs = tf.placeholder(tf.float32, [None, cube_len, cube_len, cube_len, 1], name='obs_data')
    syn = tf.placeholder(tf.float32, [None, cube_len, cube_len, cube_len, 1], name='syn_data')

    obs_res = descriptor(obs, reuse=False)
    syn_res = descriptor(syn, reuse=True)

    recon_err = tf.square(tf.reduce_mean(syn, axis=0) - tf.reduce_mean(obs, axis=0))
    des_loss = tf.subtract(tf.reduce_mean(syn_res, axis=0), tf.reduce_mean(obs_res, axis=0))

    syn_langevin = langevin_dynamics(syn)

    train_data = data_io.getObj(FLAGS.data_path, FLAGS.category, train=True, cube_len=cube_len,
                                num_voxels=FLAGS.train_size, low_bound=0, up_bound=1)
    num_voxels = len(train_data)

    incomplete_data = np.zeros(train_data.shape)
    masks = np.zeros(train_data.shape)
    for i in range(len(incomplete_data)):
        incomplete_data[i], masks[i] = get_incomplete_data(train_data[i])

    train_data = train_data[..., np.newaxis]
    incomplete_data = incomplete_data[..., np.newaxis]
    masks = masks[..., np.newaxis]

    data_io.saveVoxelsToMat(train_data, "%s/observed_data.mat" % output_dir, cmin=0, cmax=1)
    data_io.saveVoxelsToMat(incomplete_data, "%s/incomplete_data.mat" % output_dir, cmin=0, cmax=1)

    voxel_mean = train_data.mean()
    train_data = train_data - voxel_mean
    incomplete_data = incomplete_data - voxel_mean

    num_batches = int(math.ceil(num_voxels / FLAGS.batch_size))
    # descriptor variables
    des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]

    des_optim = tf.train.AdamOptimizer(FLAGS.d_lr, beta1=FLAGS.beta1)
    des_grads_vars = des_optim.compute_gradients(des_loss, var_list=des_vars)
    des_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in des_grads_vars if '/w' in var.name]
    # update by mean of gradients
    apply_d_grads = des_optim.apply_gradients(des_grads_vars)

    with tf.Session() as sess:
        # initialize training
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=50)

        recover_voxels = np.random.randn(num_voxels, cube_len, cube_len, cube_len, 1)

        des_loss_epoch = []
        recon_err_epoch = []
        plt.ion()

        for epoch in range(FLAGS.num_epochs):
            d_grad_vec = []
            des_loss_vec = []
            recon_err_vec = []

            init_data = incomplete_data.copy()
            start_time = time.time()
            for i in range(num_batches):
                indices = slice(i * FLAGS.batch_size, min(num_voxels, (i + 1) * FLAGS.batch_size))
                obs_data = train_data[indices]
                syn_data = init_data[indices]
                data_mask = masks[indices]

                # Langevin Sampling
                sample = sess.run(syn_langevin, feed_dict={syn: syn_data})
                syn_data = sample * (1 - data_mask) + syn_data * data_mask

                # learn D net
                d_grad, d_loss = \
                    sess.run([des_grads, des_loss, apply_d_grads], feed_dict={obs: obs_data, syn: syn_data})[:2]

                d_grad_vec.append(d_grad)
                des_loss_vec.append(d_loss)
                # Compute MSE
                mse = sess.run(recon_err, feed_dict={obs: obs_data, syn: syn_data})
                recon_err_vec.append(mse)
                recover_voxels[indices] = syn_data

            end_time = time.time()
            d_grad_mean, des_loss_mean, recon_err_mean = float(np.mean(d_grad_vec)), float(np.mean(des_loss_vec)), \
                                                         float(np.mean(recon_err_vec))
            des_loss_epoch.append(des_loss_mean)
            recon_err_epoch.append(recon_err_mean)
            print('Epoch #%d, descriptor loss: %.4f, descriptor SSD weight: %.4f, Avg MSE: %4.4f, time: %.2fs'
                  % (epoch, des_loss_mean, d_grad_mean, recon_err_mean,
                     end_time - start_time))

            if epoch % FLAGS.log_step == 0:
                if not os.path.exists(synthesis_dir):
                    os.makedirs(synthesis_dir)
                data_io.saveVoxelsToMat(recover_voxels + voxel_mean, "%s/sample%04d.mat" % (synthesis_dir, epoch))

                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver.save(sess, "%s/%s" % (checkpoint_dir, 'model.ckpt'), global_step=epoch)

                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                plt.figure(1)
                data_io.draw_graph(plt, des_loss_epoch, 'des_loss', log_dir, 'r')
                plt.figure(2)
                data_io.draw_graph(plt, recon_err_epoch, 'recon_error', log_dir, 'b')

def test():
    assert FLAGS.ckpt != None, 'no model provided.'
    cube_len = FLAGS.cube_len
    incomp_dir = os.path.join(FLAGS.incomp_data_path, FLAGS.category)
    test_dir = os.path.join(FLAGS.output_dir, FLAGS.category, 'test')

    syn = tf.placeholder(tf.float32, [None, cube_len, cube_len, cube_len, 1], name='syn_data')
    syn_res = descriptor(syn, reuse=False)
    syn_langevin = langevin_dynamics(syn)

    train_data = data_io.getObj(FLAGS.data_path, FLAGS.category, train=True, cube_len=cube_len,
                                num_voxels=FLAGS.train_size, low_bound=0, up_bound=1)

    incomplete_data = data_io.getVoxelsFromMat('%s/incomplete_test.mat' % incomp_dir, data_name='voxels')
    masks = np.array(io.loadmat(('%s/masks.mat' % incomp_dir))['masks'], dtype=np.float32)

    sample_size = len(incomplete_data)

    masks = masks[..., np.newaxis]
    incomplete_data = incomplete_data[..., np.newaxis]
    voxel_mean = train_data.mean()
    incomplete_data = incomplete_data - voxel_mean
    num_batches = int(math.ceil(sample_size / FLAGS.batch_size))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('Loading checkpoint {}.'.format(FLAGS.ckpt))
        saver.restore(sess, FLAGS.ckpt)

        init_data = incomplete_data.copy()
        sample_voxels = np.random.randn(sample_size, cube_len, cube_len, cube_len, 1)

        for i in range(num_batches):
            indices = slice(i * FLAGS.batch_size, min(sample_size, (i + 1) * FLAGS.batch_size))
            syn_data = init_data[indices]
            data_mask = masks[indices]

            # Langevin Sampling
            sample = sess.run(syn_langevin, feed_dict={syn: syn_data})

            sample_voxels[indices] = sample * (1 - data_mask) + syn_data * data_mask

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        data_io.saveVoxelsToMat(sample_voxels + voxel_mean, "%s/recovery.mat" % test_dir, cmin=0, cmax=1)


if __name__ == '__main__':
    if FLAGS.test:
        test()
    else:
        train()
