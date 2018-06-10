from __future__ import division
from __future__ import print_function

import tensorflow as tf
from util.custom_ops import conv3d, fully_connected


class DescriptorNet3D(object):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.cube_len = config.cube_len
        self.sample_batch = config.sample_batch
        self.sample_steps = config.sample_steps
        self.num_batches = config.num_batches

        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.step_size = config.step_size
        self.refsig = config.refsig

        self.num_epochs = config.num_epochs
        self.log_step = config.log_step

        self.syn = tf.placeholder(shape=[self.sample_batch, self.cube_len, self.cube_len, self.cube_len, 1],
                                  dtype=tf.float32)
        self.obs = tf.placeholder(shape=[None, self.cube_len, self.cube_len, self.cube_len, 1], dtype=tf.float32)

    def descriptor(self, inputs, reuse=False):
        with tf.variable_scope('des', reuse=reuse):
            # 32 x 32 x 32
            conv1 = conv3d(inputs, 200, kernal=(16, 16, 16), strides=(3, 3, 3), padding="SAME", name="conv1")
            conv1 = tf.nn.relu(conv1)

            # 12 x 12 x 12
            conv2 = conv3d(conv1, 100, kernal=(6, 6, 6), strides=(2, 2, 2), padding="SAME", name="conv2")
            conv2 = tf.nn.relu(conv2)

            # 6 x 6 x 6
            conv3 = fully_connected(conv2, 1, name="fc")
            return conv3

    def langevin_dynamics(self, syn_arg, with_noise=False):
        def cond(i, syn):
            return tf.less(i, self.sample_steps)

        def body(i, syn):
            noise = tf.random_normal(shape=tf.shape(syn), name='noise')
            syn_res = self.descriptor(syn, reuse=True)
            grad = tf.gradients(syn_res, syn, name='grad_des')[0]
            syn = syn - 0.5 * self.step_size * self.step_size * (syn / self.refsig / self.refsig - grad)
            if with_noise:
                syn = syn + self.step_size * noise
            return tf.add(i, 1), syn

        with tf.name_scope("langevin_dynamics"):
            i = tf.constant(0)
            i, syn = tf.while_loop(cond, body, [i, syn_arg])
            return syn

    def build_model(self):
        obs_res = self.descriptor(self.obs, reuse=False)
        syn_res = self.descriptor(self.syn, reuse=True)
        sample_loss = tf.reduce_sum(syn_res)
        self.sample_loss_mean, self.sample_loss_update = tf.contrib.metrics.streaming_mean(sample_loss)

        self.recon_err_mean, self.recon_err_update = tf.contrib.metrics.streaming_mean_squared_error(
            tf.reduce_mean(self.syn, axis=0), tf.reduce_mean(self.obs, axis=0))

        self.langevin_descriptor = self.langevin_dynamics(self.syn, False)
        self.langevin_descriptor_noise = self.langevin_dynamics(self.syn, True)

        # descriptor variables
        des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]
        accum_des_vars = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in des_vars]

        # initialize training

        des_loss = tf.subtract(tf.reduce_mean(syn_res, axis=0), tf.reduce_mean(obs_res, axis=0))
        self.des_loss_mean, self.des_loss_update = tf.contrib.metrics.streaming_mean(des_loss)

        des_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1)
        des_grads_vars = des_optim.compute_gradients(des_loss, var_list=des_vars)
        self.des_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in des_grads_vars if '/w' in var.name]
        self.update_d_grads = [accum_des_vars[i].assign_add(gv[0]) for i, gv in enumerate(des_grads_vars)]
        # update by mean of gradients
        self.apply_d_grads = des_optim.apply_gradients([(tf.divide(accum_des_vars[i], self.num_batches), gv[1])
                                                   for i, gv in enumerate(des_grads_vars)])

        self.reset_grads = [var.assign(tf.zeros_like(var)) for var in accum_des_vars]

        tf.summary.scalar('des_loss', self.des_loss_mean)
        tf.summary.scalar('sample_loss', self.sample_loss_mean)
        tf.summary.scalar('recon_err', self.recon_err_mean)

        self.summary_op = tf.summary.merge_all()