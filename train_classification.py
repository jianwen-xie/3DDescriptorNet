from __future__ import division
from __future__ import print_function

import os
import math
from config import FLAGS

from util import data_io
from util.custom_ops import *
from liblinear.python.liblinearutil import *

tf.flags.DEFINE_integer('num_class', 10, 'Number of classes of 3D objects.')
tf.flags.DEFINE_string('classifier_type', 'logistic', 'Support logistic or svm')


def extract_features(inputs, reuse=False):
    with tf.variable_scope('des', reuse=reuse):
        conv1 = conv3d(inputs, 200, kernal=(16, 16, 16), strides=(3, 3, 3), padding="SAME", name="conv1")
        conv1 = tf.nn.relu(conv1)
        conv2 = conv3d(conv1, 100, kernal=(6, 6, 6), strides=(2, 2, 2), padding="SAME", name="conv2")
        conv2 = tf.nn.relu(conv2)

    conv1_mp = tf.layers.max_pooling3d(conv1, pool_size=[4, 4, 4], strides=[4, 4, 4], padding="SAME")
    conv2_mp = tf.layers.max_pooling3d(conv2, pool_size=[2, 2, 2], strides=[2, 2, 2], padding="SAME")
    features = tf.concat([tf.layers.flatten(conv1_mp), tf.layers.flatten(conv2_mp)], axis=1)

    return features


def discriminator(inputs, reuse=False):
    with tf.variable_scope('dis', reuse=reuse):
        conv3_1 = tf.layers.flatten(inputs)
        logits = tf.layers.dense(conv3_1, FLAGS.num_class)
        return logits


def train_svm():
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.category)
    log_dir = os.path.join(output_dir, 'log')

    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    # Prepare training data
    train_data, train_labels = data_io.getAll(FLAGS.data_path, cube_len=FLAGS.cube_len, low_bound=0, up_bound=1,
                                                  num_voxels=FLAGS.train_size)
    test_data, test_labels = data_io.getAll(FLAGS.data_path, cube_len=FLAGS.cube_len, low_bound=0, up_bound=1,
                                                num_voxels=FLAGS.test_size, train=False)
    voxel_mean = train_data.mean()
    train_data = train_data - voxel_mean
    test_data = test_data - voxel_mean
    train_data = train_data[..., np.newaxis]
    test_data = test_data[..., np.newaxis]
    print('Reading voxel data, shape: {}'.format(train_data.shape))
    print('min: %.4f\tmax: %.4f' % (train_data.min(), train_data.max()))

    obs = tf.placeholder(shape=[None, FLAGS.cube_len, FLAGS.cube_len, FLAGS.cube_len, 1], dtype=tf.float32)

    features = extract_features(obs)

    saver = tf.train.Saver(tf.trainable_variables(scope='des'))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.ckpt)

        print('Loading checkpoint {}'.format(FLAGS.ckpt))

        num_batches_train = int(math.ceil(len(train_data) / FLAGS.batch_size))
        num_batches_test = int(math.ceil(len(test_data) / FLAGS.batch_size))

        train_features = []
        test_features = []

        # Extract features of training and testing datasets
        for i in range(num_batches_train):
            train_x = train_data[i * FLAGS.batch_size:min(len(train_data), (i + 1) * FLAGS.batch_size)]
            ff = sess.run(features, feed_dict={obs: train_x})
            train_features.append(ff)
        train_features = np.concatenate(train_features, axis=0)
        print(train_features.shape)

        for i in range(num_batches_test):
            test_x = test_data[i * FLAGS.batch_size:min(len(test_data), (i + 1) * FLAGS.batch_size)]
            ff = sess.run(features, feed_dict={obs: test_x})
            test_features.append(ff)
        test_features = np.concatenate(test_features, axis=0)

        # train SVM
        print('Begin to train SVM .........')
        prob = problem(train_labels, train_features)
        param = parameter('-s 2 -c 0.01')
        svm_model = train(prob, param)
        _, train_acc, _ = predict(train_labels, train_features, svm_model)
        _, test_acc, _ = predict(test_labels, test_features, svm_model)
        print('train acc: %.4f, test acc: %.4f' % (train_acc[0], test_acc[0]))


def train_logistic():
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.category)
    log_dir = os.path.join(output_dir, 'log')
    model_dir = os.path.join(output_dir, 'model')

    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    tf.gfile.MakeDirs(model_dir)

    obs = tf.placeholder(shape=[None, FLAGS.cube_len, FLAGS.cube_len, FLAGS.cube_len, 1], dtype=tf.float32)
    labels = tf.placeholder(shape=[None], dtype=tf.int64)

    # Prepare checking point
    des_res = extract_features(obs, reuse=False)
    dis_res = discriminator(des_res, reuse=False)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=dis_res)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(dis_res, 1), labels), tf.float32))

    optim = tf.train.AdamOptimizer(FLAGS.d_lr)
    t_vars = tf.trainable_variables(scope='dis')

    apply_grads = optim.minimize(loss, var_list=t_vars)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', accuracy)

    # Prepare training data
    train_data, train_labels = data_io.getAll(FLAGS.data_path, cube_len=FLAGS.cube_len, low_bound=0, up_bound=1,
                                              num_voxels=FLAGS.train_size)
    test_data, test_labels = data_io.getAll(FLAGS.data_path, cube_len=FLAGS.cube_len, low_bound=0, up_bound=1,
                                            num_voxels=FLAGS.test_size, train=False)
    voxel_mean = train_data.mean()
    train_data = train_data - voxel_mean
    test_data = test_data - voxel_mean
    train_data = train_data[..., np.newaxis]
    test_data = test_data[..., np.newaxis]
    print('Reading voxel data, shape: {}'.format(train_data.shape))
    print('min: %.4f\tmax: %.4f' % (train_data.min(), train_data.max()))

    acc_max_train = 0
    acc_max_test = 0

    # begin training
    num_batches_train = int(math.ceil(len(train_data) / FLAGS.batch_size))
    num_batches_test = int(math.ceil(len(test_data) / FLAGS.batch_size))

    saver = tf.train.Saver(tf.trainable_variables(scope='des'))
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.ckpt)
        sess.graph.finalize()

        writer = tf.summary.FileWriter(log_dir, sess.graph)

        print('Loading checkpoint {}'.format(FLAGS.ckpt))

        for epoch in range(FLAGS.num_epochs):
            shuffle_idx = np.random.permutation(np.arange(len(train_data)))
            train_data = train_data[shuffle_idx]
            train_labels = train_labels[shuffle_idx]
            Acc = []
            Loss = []
            for i in range(num_batches_train):
                train_x = train_data[i * FLAGS.batch_size:min(len(train_data), (i + 1) * FLAGS.batch_size)]
                train_y = train_labels[i * FLAGS.batch_size:min(len(train_data), (i + 1) * FLAGS.batch_size)]

                d_res, acc, l = sess.run([dis_res, accuracy, loss, apply_grads],
                                              feed_dict={obs: train_x, labels: train_y})[:3]
                # print dis_res
                Acc.append(acc)
                Loss.append(l)

            Acc_test = []
            Loss_test = []
            for i in range(num_batches_test):
                test_x = test_data[i * FLAGS.batch_size:min(len(test_data), (i + 1) * FLAGS.batch_size)]
                test_y = test_labels[i * FLAGS.batch_size:min(len(test_data), (i + 1) * FLAGS.batch_size)]
                acc, l = sess.run([accuracy, loss], feed_dict={obs: test_x, labels: test_y})

                Acc_test.append(acc)
                Loss_test.append(l)

            if acc_max_test < np.mean(Acc_test):
                acc_max_test = np.mean(Acc_test)
                acc_max_train = np.mean(Acc)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                saver.save(sess, "%s/%s" % (model_dir, 'model.ckpt'))

            summary = sess.run(summary_op, feed_dict={obs: test_x, labels: test_y})
            writer.add_summary(summary, epoch)

            print('Epoch #%d, train loss: %.4f, train acc: %.4f, test loss: %.4f, test acc: %.4f'
                  % (epoch, float(np.mean(Loss)), float(acc_max_train), float(np.mean(Loss_test)), float(acc_max_test)))


def main(_):
    # Prepare checking point
    assert FLAGS.ckpt is not None, 'no checkpoint provided.'

    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)

    if FLAGS.classifier_type == 'svm':
        train_svm()
    elif FLAGS.classifier_type == 'logistic':
        train_logistic()
    else:
        return NotImplementedError


if __name__ == '__main__':
    tf.app.run(main)
