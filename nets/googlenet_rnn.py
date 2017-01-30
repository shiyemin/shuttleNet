from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ops import fops
from ops.network import Network
import tensorflow as tf

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

class GoogleNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, name='conv1_7x7_s2')
             .max_pool(3, 3, 2, 2, name='pool1_3x3_s2')
             .lrn(2, 2e-05, 0.75, name='pool1_norm1')
             .conv(1, 1, 64, 1, 1, name='conv2_3x3_reduce')
             .conv(3, 3, 192, 1, 1, name='conv2_3x3')
             .lrn(2, 2e-05, 0.75, name='conv2_norm2')
             .max_pool(3, 3, 2, 2, name='pool2_3x3_s2')
             .conv(1, 1, 64, 1, 1, name='inception_3a_1x1'))

        (self.feed('pool2_3x3_s2')
             .conv(1, 1, 96, 1, 1, name='inception_3a_3x3_reduce')
             .conv(3, 3, 128, 1, 1, name='inception_3a_3x3'))

        (self.feed('pool2_3x3_s2')
             .conv(1, 1, 16, 1, 1, name='inception_3a_5x5_reduce')
             .conv(5, 5, 32, 1, 1, name='inception_3a_5x5'))

        (self.feed('pool2_3x3_s2')
             .max_pool(3, 3, 1, 1, name='inception_3a_pool')
             .conv(1, 1, 32, 1, 1, name='inception_3a_pool_proj'))

        (self.feed('inception_3a_1x1',
                   'inception_3a_3x3',
                   'inception_3a_5x5',
                   'inception_3a_pool_proj')
             .concat(3, name='inception_3a_output')
             .conv(1, 1, 128, 1, 1, name='inception_3b_1x1'))

        (self.feed('inception_3a_output')
             .conv(1, 1, 128, 1, 1, name='inception_3b_3x3_reduce')
             .conv(3, 3, 192, 1, 1, name='inception_3b_3x3'))

        (self.feed('inception_3a_output')
             .conv(1, 1, 32, 1, 1, name='inception_3b_5x5_reduce')
             .conv(5, 5, 96, 1, 1, name='inception_3b_5x5'))

        (self.feed('inception_3a_output')
             .max_pool(3, 3, 1, 1, name='inception_3b_pool')
             .conv(1, 1, 64, 1, 1, name='inception_3b_pool_proj'))

        (self.feed('inception_3b_1x1',
                   'inception_3b_3x3',
                   'inception_3b_5x5',
                   'inception_3b_pool_proj')
             .concat(3, name='inception_3b_output')
             .max_pool(3, 3, 2, 2, name='pool3_3x3_s2')
             .conv(1, 1, 192, 1, 1, name='inception_4a_1x1'))

        (self.feed('pool3_3x3_s2')
             .conv(1, 1, 96, 1, 1, name='inception_4a_3x3_reduce')
             .conv(3, 3, 208, 1, 1, name='inception_4a_3x3'))

        (self.feed('pool3_3x3_s2')
             .conv(1, 1, 16, 1, 1, name='inception_4a_5x5_reduce')
             .conv(5, 5, 48, 1, 1, name='inception_4a_5x5'))

        (self.feed('pool3_3x3_s2')
             .max_pool(3, 3, 1, 1, name='inception_4a_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4a_pool_proj'))

        (self.feed('inception_4a_1x1',
                   'inception_4a_3x3',
                   'inception_4a_5x5',
                   'inception_4a_pool_proj')
             .concat(3, name='inception_4a_output')
             .conv(1, 1, 160, 1, 1, name='inception_4b_1x1'))

        (self.feed('inception_4a_output')
             .conv(1, 1, 112, 1, 1, name='inception_4b_3x3_reduce')
             .conv(3, 3, 224, 1, 1, name='inception_4b_3x3'))

        (self.feed('inception_4a_output')
             .conv(1, 1, 24, 1, 1, name='inception_4b_5x5_reduce')
             .conv(5, 5, 64, 1, 1, name='inception_4b_5x5'))

        (self.feed('inception_4a_output')
             .max_pool(3, 3, 1, 1, name='inception_4b_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4b_pool_proj'))

        (self.feed('inception_4b_1x1',
                   'inception_4b_3x3',
                   'inception_4b_5x5',
                   'inception_4b_pool_proj')
             .concat(3, name='inception_4b_output')
             .conv(1, 1, 128, 1, 1, name='inception_4c_1x1'))

        (self.feed('inception_4b_output')
             .conv(1, 1, 128, 1, 1, name='inception_4c_3x3_reduce')
             .conv(3, 3, 256, 1, 1, name='inception_4c_3x3'))

        (self.feed('inception_4b_output')
             .conv(1, 1, 24, 1, 1, name='inception_4c_5x5_reduce')
             .conv(5, 5, 64, 1, 1, name='inception_4c_5x5'))

        (self.feed('inception_4b_output')
             .max_pool(3, 3, 1, 1, name='inception_4c_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4c_pool_proj'))

        (self.feed('inception_4c_1x1',
                   'inception_4c_3x3',
                   'inception_4c_5x5',
                   'inception_4c_pool_proj')
             .concat(3, name='inception_4c_output')
             .conv(1, 1, 112, 1, 1, name='inception_4d_1x1'))

        (self.feed('inception_4c_output')
             .conv(1, 1, 144, 1, 1, name='inception_4d_3x3_reduce')
             .conv(3, 3, 288, 1, 1, name='inception_4d_3x3'))

        (self.feed('inception_4c_output')
             .conv(1, 1, 32, 1, 1, name='inception_4d_5x5_reduce')
             .conv(5, 5, 64, 1, 1, name='inception_4d_5x5'))

        (self.feed('inception_4c_output')
             .max_pool(3, 3, 1, 1, name='inception_4d_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4d_pool_proj'))

        (self.feed('inception_4d_1x1',
                   'inception_4d_3x3',
                   'inception_4d_5x5',
                   'inception_4d_pool_proj')
             .concat(3, name='inception_4d_output')
             .conv(1, 1, 256, 1, 1, name='inception_4e_1x1'))

        (self.feed('inception_4d_output')
             .conv(1, 1, 160, 1, 1, name='inception_4e_3x3_reduce')
             .conv(3, 3, 320, 1, 1, name='inception_4e_3x3'))

        (self.feed('inception_4d_output')
             .conv(1, 1, 32, 1, 1, name='inception_4e_5x5_reduce')
             .conv(5, 5, 128, 1, 1, name='inception_4e_5x5'))

        (self.feed('inception_4d_output')
             .max_pool(3, 3, 1, 1, name='inception_4e_pool')
             .conv(1, 1, 128, 1, 1, name='inception_4e_pool_proj'))

        (self.feed('inception_4e_1x1',
                   'inception_4e_3x3',
                   'inception_4e_5x5',
                   'inception_4e_pool_proj')
             .concat(3, name='inception_4e_output')
             .max_pool(3, 3, 2, 2, name='pool4_3x3_s2')
             .conv(1, 1, 256, 1, 1, name='inception_5a_1x1'))

        (self.feed('pool4_3x3_s2')
             .conv(1, 1, 160, 1, 1, name='inception_5a_3x3_reduce')
             .conv(3, 3, 320, 1, 1, name='inception_5a_3x3'))

        (self.feed('pool4_3x3_s2')
             .conv(1, 1, 32, 1, 1, name='inception_5a_5x5_reduce')
             .conv(5, 5, 128, 1, 1, name='inception_5a_5x5'))

        (self.feed('pool4_3x3_s2')
             .max_pool(3, 3, 1, 1, name='inception_5a_pool')
             .conv(1, 1, 128, 1, 1, name='inception_5a_pool_proj'))

        (self.feed('inception_5a_1x1',
                   'inception_5a_3x3',
                   'inception_5a_5x5',
                   'inception_5a_pool_proj')
             .concat(3, name='inception_5a_output')
             .conv(1, 1, 384, 1, 1, name='inception_5b_1x1'))

        (self.feed('inception_5a_output')
             .conv(1, 1, 192, 1, 1, name='inception_5b_3x3_reduce')
             .conv(3, 3, 384, 1, 1, name='inception_5b_3x3'))

        (self.feed('inception_5a_output')
             .conv(1, 1, 48, 1, 1, name='inception_5b_5x5_reduce')
             .conv(5, 5, 128, 1, 1, name='inception_5b_5x5'))

        (self.feed('inception_5a_output')
             .max_pool(3, 3, 1, 1, name='inception_5b_pool')
             .conv(1, 1, 128, 1, 1, name='inception_5b_pool_proj'))

        (self.feed('inception_5b_1x1',
                   'inception_5b_3x3',
                   'inception_5b_5x5',
                   'inception_5b_pool_proj')
             .concat(3, name='inception_5b_output')
             .avg_pool(7, 7, 1, 1, padding='VALID', name='pool5_7x7_s1'))


def googlenet_rnn(images, num_classes=101, is_training=True,
          dropout_keep_prob=0.4,
          prediction_fn=slim.softmax,
          scope='GoogleNet'):
    """Creates a variant of the GoogLeNet model.

    Note that since the output is a set of 'logits', the values fall in the
    interval of (-infinity, infinity). Consequently, to convert the outputs to a
    probability distribution over the characters, one will need to convert them
    using the softmax function:

            logits = googlenet.googlenet(images, is_training=False)
            probabilities = tf.nn.softmax(logits)
            predictions = tf.argmax(logits, 1)

    Args:
        images: A batch of `Tensors` of size [batch_size, height, width, channels].
        num_classes: the number of classes in the dataset.
        is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
        dropout_keep_prob: the percentage of activation values that are retained.
        prediction_fn: a function to get predictions out of logits.
        scope: Optional variable_scope.

    Returns:
        logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
        end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    end_points = {}
    network = GoogleNet({'data':images}, trainable=is_training)
    net = network.get_output()

    with tf.variable_scope('shuttleNet'):
        net = slim.flatten(net)
        end_points['Flatten'] = net

        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                        scope='dropout')
        net = fops.shuttleNet(net, FLAGS.n_steps, 2, 1024, 2,
                            num_frames=None, echocell="GRU",
                            scope='shuttleNet')
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                             normalizer_fn=None, scope='FC_prediction')

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points
googlenet_rnn.default_image_size = 224


def googlenet_rnn_arg_scope(weight_decay=0.0002):
    """Defines the default googlenet argument scope.

    Args:
        weight_decay: The weight decay to use for regularizing the model.

    Returns:
        An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu) as sc:
            return sc
