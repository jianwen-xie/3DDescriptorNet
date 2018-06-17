import tensorflow as tf

# training hyper-parameters
tf.flags.DEFINE_integer('num_epochs', 3000, 'Number of epochs to train')
tf.flags.DEFINE_integer('batch_size', 20, 'Batch size of training data')
tf.flags.DEFINE_integer('cube_len', 32, 'Volumetric cube size')
tf.flags.DEFINE_float('d_lr', 0.01, 'Learning rate for descriptor')
tf.flags.DEFINE_float('beta1', 0.5, 'Momentum1 in Adam')
tf.flags.DEFINE_float('step_size', 0.1, 'Step size for descriptor Langevin dynamics')
tf.flags.DEFINE_float('refsig', 0.5, 'Standard deviation for reference distribution of descriptor')
tf.flags.DEFINE_integer('sample_steps', 20, 'Sample steps for Langevin dynamics of descriptor')
tf.flags.DEFINE_integer('train_size', None, 'Number of maximum training data')
tf.flags.DEFINE_integer('log_step', 10, 'Number of epochs to save output results')

# testing hyper-parameters
tf.flags.DEFINE_boolean('test', False, 'True if in testing mode')
tf.flags.DEFINE_string('ckpt', None, 'Checkpoint path to load')
tf.flags.DEFINE_integer('test_size', None, 'Number of maximum testing data')

# data hyper-parameters
tf.flags.DEFINE_string('category', 'toilet', 'Name of subcategory in dataset')
tf.flags.DEFINE_string('data_path', './data/volumetric_data/ModelNet10', 'The dataset root directory')
tf.flags.DEFINE_string('output_dir', './output', 'The output directory for saving results')

FLAGS = tf.flags.FLAGS