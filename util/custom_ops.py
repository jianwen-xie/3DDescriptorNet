import numpy as np
import tensorflow as tf


def leaky_relu(input_, leakiness=0.2):
    assert leakiness <= 1
    return tf.maximum(input_, leakiness * input_)


def fully_connected(input_, output_dim, name="fc"):
    shape = input_.shape
    return conv3d(input_, output_dim, kernal=list(shape[1:4]), strides=(1, 1, 1), padding="VALID", name=name)


def up_sample(input_, scale=4, name="up_sample"):
    with tf.variable_scope(name):
        w = tf.Variable(tf.constant(1, shape=(1, 1, 1, 1, 1)), name="w")
        return tf.nn.conv3d_transpose(input_, w, output_shape=(), strides=scale, padding="VALID")


def convt3d(input_, output_shape, kernal=(5, 5, 5), strides=(2, 2, 2), padding='SAME', activation_fn=None,
            name="convt3d"):
    assert type(kernal) in [list, tuple, int]
    assert type(strides) in [list, tuple, int]
    assert type(padding) in [list, tuple, int, str]
    if type(kernal) == list or type(kernal) == tuple:
        [k_d, k_h, k_w] = list(kernal)
    else:
        k_d = k_h = k_w = kernal
    if type(strides) == list or type(strides) == tuple:
        [d_d, d_h, d_w] = list(strides)
    else:
        d_d = d_h = d_w = strides
    output_shape = list(output_shape)
    output_shape[0] = tf.shape(input_)[0]
    with tf.variable_scope(name):
        if type(padding) in [tuple, list, int]:
            if type(padding) == int:
                p_d = p_h = p_w = padding
            else:
                [p_d, p_h, p_w] = list(padding)
            pad_ = [0, p_d, p_h, p_w, 0]
            input_ = tf.pad(input_, [[p, p] for p in pad_], "CONSTANT")
            padding = 'VALID'

        w = tf.get_variable('w', [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=0.001))
        convt = tf.nn.conv3d_transpose(input_, w, output_shape=tf.stack(output_shape, axis=0),
                                       strides=[1, d_d, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.bias_add(convt, biases)
        if activation_fn != None:
            convt = activation_fn(convt)
        return convt


def conv3d(input_, output_dim, kernal=(5, 5, 5), strides=(2, 2, 2), padding='SAME', activation_fn=None, name="conv3d"):
    if type(kernal) == list or type(kernal) == tuple:
        [k_d, k_h, k_w] = list(kernal)
    else:
        k_d = k_h = k_w = kernal
    if type(strides) == list or type(strides) == tuple:
        [d_d, d_h, d_w] = list(strides)
    else:
        d_d = d_h = d_w = strides

    with tf.variable_scope(name):

        if type(padding) == list or type(padding) == tuple:
            padding = [0] + list(padding) + [0]
            input_ = tf.pad(input_, [[p, p] for p in padding], "CONSTANT")
            padding = 'VALID'
        w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=0.001))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        if activation_fn != None:
            conv = activation_fn(conv)
        return conv
