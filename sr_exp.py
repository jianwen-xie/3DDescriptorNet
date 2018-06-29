from __future__ import division

import math
import os
import time
from config import FLAGS
from util import data_io
from util.custom_ops import *

tf.flags.DEFINE_integer('scale', 4, 'Upsampling scale for super resolution.')

def upsample(input_, up_scale):
    with tf.variable_scope("upsample"):
        return tf.keras.layers.UpSampling3D(size=(up_scale, up_scale, up_scale), dtype=tf.float32)(input_)

def downsample(input_, down_scale):
    with tf.variable_scope("downsample"):
        return tf.keras.layers.AveragePooling3D(pool_size=(down_scale, down_scale, down_scale), padding='valid',
                                                dtype=tf.float32)(input_)

def avg_pool(input_, scale):
    with tf.variable_scope("avg_pool"):
        ls = downsample(input_, scale)
        hs = upsample(ls, scale)
        return hs

def descriptor(inputs, reuse=False):
    with tf.variable_scope('des', reuse=reuse):
        # 64
        conv1 = conv3d(inputs, 200, kernal=(16, 16, 16), strides=(3, 3, 3), padding="SAME", name="conv1")
        conv1 = tf.nn.relu(conv1)
        print conv1

        fc = fully_connected(conv1, 1, name="fc")
        print fc

        return fc

def langevin_dynamics(syn_arg):
    def cond(i, syn):
        return tf.less(i, FLAGS.sample_steps)

    def body(i, syn):
        syn_res = descriptor(syn, reuse=True)
        grad = tf.gradients(syn_res, syn, name='grad_des')[0]
        y = syn - 0.5 * FLAGS.step_size * FLAGS.step_size * (syn / FLAGS.refsig / FLAGS.refsig - grad)
        syn =  y - avg_pool(y - syn, FLAGS.scale)
        return tf.add(i, 1), syn

    with tf.name_scope("langevin_dynamics"):
        i = tf.constant(0)
        i, syn = tf.while_loop(cond, body, [i, syn_arg])
        return syn


def train():
    cube_len = FLAGS.cube_len
    scale = FLAGS.scale
    batch_size = FLAGS.batch_size

    output_dir = os.path.join(FLAGS.output_dir, FLAGS.category)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    synthesis_dir = os.path.join(output_dir, 'sr_results')
    log_dir = os.path.join(output_dir, 'log')

    lr_size = cube_len // scale
    obs = tf.placeholder(tf.float32, [None, cube_len, cube_len, cube_len, 1], name='obs_data')
    syn = tf.placeholder(tf.float32, [None, cube_len, cube_len, cube_len, 1], name='syn_data')
    low_res = tf.placeholder(tf.float32, [None, lr_size, lr_size, lr_size, 1], name='low_res')

    down_syn = downsample(obs, scale)
    up_syn = upsample(low_res, scale)
    obs_res = descriptor(obs, reuse=False)
    syn_res = descriptor(syn, reuse=True)
    sr_res = obs + syn - avg_pool(syn, scale)

    recon_err_mean, recon_err_update = tf.contrib.metrics.streaming_mean_squared_error(
        tf.reduce_mean(syn, axis=0), tf.reduce_mean(obs, axis=0))

    des_loss = tf.subtract(tf.reduce_mean(syn_res, axis=0), tf.reduce_mean(obs_res, axis=0))
    des_loss_mean, des_loss_update = tf.contrib.metrics.streaming_mean(des_loss)
    dLdI = tf.gradients(syn_res, syn)[0]

    syn_langevin = langevin_dynamics(syn)

    tf.summary.scalar('des_loss', des_loss_mean)
    tf.summary.scalar('recon_err', recon_err_mean)

    train_data = data_io.getObj(FLAGS.data_path, FLAGS.category, cube_len=cube_len, num_voxels=FLAGS.train_size)
    num_voxels = len(train_data)

    train_data = train_data[..., np.newaxis]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_io.saveVoxelsToMat(train_data, "%s/observed_data.mat" % output_dir, cmin=0, cmax=1)


    num_batches = int(math.ceil(num_voxels / batch_size))
    # descriptor variables
    des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]

    des_optim = tf.train.AdamOptimizer(FLAGS.d_lr, beta1=FLAGS.beta1)
    des_grads_vars = des_optim.compute_gradients(des_loss, var_list=des_vars)
    des_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in des_grads_vars if '/w' in var.name]
    apply_d_grads = des_optim.apply_gradients(des_grads_vars)

    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=50)

    with tf.Session() as sess:
        # initialize training
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        writer = tf.summary.FileWriter(log_dir, sess.graph)

        print('Generating low resolution data')
        up_samples = np.zeros(train_data.shape)
        down_samples = np.zeros(shape=[num_voxels, lr_size, lr_size, lr_size, 1])
        for i in range(num_batches):
            indices = slice(i * batch_size, min(num_voxels, (i + 1) * batch_size))
            obs_data = train_data[indices]
            ds = sess.run(down_syn, feed_dict={obs: obs_data})
            us = sess.run(up_syn, feed_dict={low_res: ds})
            down_samples[indices] = ds
            up_samples[indices] = us

        data_io.saveVoxelsToMat(down_samples, "%s/down_sample.mat" % output_dir, cmin=0, cmax=1)
        data_io.saveVoxelsToMat(up_samples, "%s/up_sample.mat" % output_dir, cmin=0, cmax=1)

        voxel_mean = train_data.mean()
        train_data = train_data - voxel_mean
        up_samples = up_samples - voxel_mean

        print('start training')
        sample_voxels = np.random.randn(num_voxels, cube_len, cube_len, cube_len, 1)

        for epoch in range(FLAGS.num_epochs):

            start_time = time.time()
            for i in range(num_batches):
                indices = slice(i * batch_size, min(num_voxels, (i + 1) * batch_size))
                obs_data = train_data[indices]
                us_data = up_samples[indices]

                sr = sess.run(syn_langevin, feed_dict={syn: us_data})

                # learn D net
                sess.run([des_loss_update, apply_d_grads], feed_dict={obs: obs_data, syn: sr})
                # Compute MSE
                sess.run(recon_err_update, feed_dict={obs: obs_data, syn: sr})

                sample_voxels[indices] = sr

            [des_loss_avg, mse, summary] = sess.run([des_loss_mean, recon_err_mean, summary_op])
            end_time = time.time()
            print('Epoch #%d, descriptor loss: %.4f, avg MSE: %4.4f, time:%.2fs'
                  % (epoch, des_loss_avg, mse, end_time-start_time))
            writer.add_summary(summary, epoch)

            if mse > 2 or np.isnan(mse):
                break

            if epoch % FLAGS.log_step == 0:
                if not os.path.exists(synthesis_dir):
                    os.makedirs(synthesis_dir)
                data_io.saveVoxelsToMat(sample_voxels + voxel_mean, "%s/sample%04d.mat" % (synthesis_dir, epoch), cmin=0, cmax=1)

                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver.save(sess, "%s/%s" % (checkpoint_dir, 'model.ckpt'), global_step=epoch)

def test():
    assert FLAGS.ckpt != None, 'no model provided.'
    cube_len = FLAGS.cube_len
    scale = FLAGS.scale
    batch_size = FLAGS.batch_size

    test_dir = os.path.join(FLAGS.output_dir, FLAGS.category, 'test')

    lr_size = cube_len // scale
    obs = tf.placeholder(tf.float32, [None, cube_len, cube_len, cube_len, 1], name='obs_data')
    syn = tf.placeholder(tf.float32, [None, cube_len, cube_len, cube_len, 1], name='syn_data')
    low_res = tf.placeholder(tf.float32, [None, lr_size, lr_size, lr_size, 1], name='low_res')

    down_syn = downsample(obs, scale)
    up_syn = upsample(low_res, scale)

    syn_res = descriptor(syn, reuse=False)
    syn_langevin = langevin_dynamics(syn)
    sr_res = obs + syn - avg_pool(syn, scale)

    train_data = data_io.getObj(FLAGS.data_path, FLAGS.category, train=True, cube_len=cube_len, num_voxels=FLAGS.train_size)
    test_data = data_io.getObj(FLAGS.data_path, FLAGS.category, train=False, cube_len=cube_len, num_voxels=FLAGS.test_size)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    data_io.saveVoxelsToMat(test_data, "%s/observed_data.mat" % test_dir, cmin=0, cmax=1)
    sample_size = len(test_data)
    num_batches = int(math.ceil(sample_size / batch_size))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('Generating low resolution data')
        test_data = test_data[..., np.newaxis]
        up_samples = np.zeros(test_data.shape)
        down_samples = np.zeros(shape=[sample_size, lr_size, lr_size, lr_size, 1])
        for i in range(num_batches):
            indices = slice(i * batch_size, min(sample_size, (i + 1) * batch_size))
            obs_data = test_data[indices]
            ds = sess.run(down_syn, feed_dict={obs: obs_data})
            us = sess.run(up_syn, feed_dict={low_res: ds})
            down_samples[indices] = ds
            up_samples[indices] = us

        data_io.saveVoxelsToMat(down_samples, "%s/down_sample.mat" % test_dir, cmin=0, cmax=1)
        data_io.saveVoxelsToMat(up_samples, "%s/up_sample.mat" % test_dir, cmin=0, cmax=1)

        voxel_mean = train_data.mean()
        up_samples = up_samples - voxel_mean

        print 'Loading checkpoint {}.'.format(FLAGS.ckpt)
        saver.restore(sess, FLAGS.ckpt)

        init_data = up_samples.copy()
        sample_voxels = np.random.randn(sample_size, cube_len, cube_len, cube_len, 1)

        for i in range(num_batches):
            indices = slice(i * batch_size, min(sample_size, (i + 1) * batch_size))
            us_data = init_data[indices]

            # Langevin Sampling
            y1 = sess.run(syn_langevin, feed_dict={syn: us_data})

            sample_voxels[indices] = y1

        data_io.saveVoxelsToMat(sample_voxels + voxel_mean, "%s/samples.mat" % test_dir, cmin=0, cmax=1)

if __name__ == '__main__':
    if FLAGS.test:
        test()
    else:
        train()