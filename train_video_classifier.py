# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os
import math
from datetime import datetime

from tensorflow.python.ops import control_flow_ops
from deployment import model_deploy
from deployment import train_util
from nets import nets_factory
from preprocessing import preprocessing_factory

import async_loader

try:
    xrange
except NameError:
    xrange = range
slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'mode', 'train',
    'Run mode. train or test or extract.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/eval_log/',
    'Directory where event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'trace_every_n_steps', None,
    'The frequency with which the timeline is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

tf.app.flags.DEFINE_boolean(
    'log_device_placement', False,
    """Whether to log device placement.""")

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' "piecewise", or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_string(
    'learning_rate_steps',
    '10', 'Setting the exact learning rate steps when FLAGS.learning_rate is set as 0.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_string(
    'decay_iteration', '10000',
    'Number of iterations after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_float(
    'grad_clipping', None,
    """Gradient cliping by norm.""")

tf.app.flags.DEFINE_boolean(
    'no_decay', False,
    """Whether decay the learning rate of recovered variables.""")

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_list', '', 'The list of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'NUM_CLASSES', 101,
    'The number of classes.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'resize_image_size', 256, 'Train image size')

tf.app.flags.DEFINE_integer(
    'train_image_size', 224, 'Train image size')

tf.app.flags.DEFINE_integer(
    'max_number_of_steps', None,
    'The maximum number of training steps.')

tf.app.flags.DEFINE_integer('top_k', 5,
                            """Top k accuracy.""")

tf.app.flags.DEFINE_string(
    'feature_dir', '/tmp/tfmodel/',
    'Directory where features are written to.')

tf.app.flags.DEFINE_string(
    'rnn', 'shuttleNet', 'The list of the dataset to load.')

tf.app.flags.DEFINE_string(
    'echocell', 'GRUBlock', 'The list of the dataset to load.')

tf.app.flags.DEFINE_integer(
    'num_rnn', 1,
    'The Number of rnn layers.')

#####################
# Video Flags #
#####################

tf.app.flags.DEFINE_integer('n_steps', 16,
                            """Time steps for LSTM.""")

tf.app.flags.DEFINE_integer('length', 1, """Sample length.""")

tf.app.flags.DEFINE_string('modality', 'None',
                           """Modality of data.""")

tf.app.flags.DEFINE_integer('read_stride', 5,
                            """Read stride of video frames.""")

tf.app.flags.DEFINE_boolean(
    'merge_label', False,
    'If output one label for each video.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'npy_weights', None,
    'The path to a weights.npy from which to fine-tune.')

tf.app.flags.DEFINE_boolean(
    'no_restore_exclude', False,
    'Prevent checkpoint_exclude_scopes parameters.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_end_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

tf.flags.DEFINE_integer(
    "eval_interval_secs", 1200,
    "Interval between evaluation runs.")

FLAGS = tf.app.flags.FLAGS


def train():
    with tf.Graph().as_default():
        ######################
        # Config model_deploy#
        ######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        ####################
        # Select the network #
        ####################
        network_fn = nets_factory.get_network_fn(
                    FLAGS.model_name,
                    num_classes=FLAGS.NUM_CLASSES,
                    weight_decay=FLAGS.weight_decay,
                    is_training=True)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                    preprocessing_name,
                    is_training=True)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        if hasattr(network_fn, 'rnn_part'):
            load_batch_size = FLAGS.batch_size * deploy_config.num_clones
        else:
            load_batch_size = FLAGS.batch_size
        with tf.device(deploy_config.inputs_device()):
            dataset_size, images, labels, video_name = async_loader.video_inputs(FLAGS.dataset_list,
                                        FLAGS.dataset_dir, FLAGS.resize_image_size, FLAGS.train_image_size,
                                        load_batch_size, FLAGS.n_steps, FLAGS.modality, FLAGS.read_stride,
                                        image_preprocessing_fn, shuffle=True,
                                        label_from_one=(FLAGS.labels_offset>0),
                                        length1=FLAGS.length, crop=2,
                                        merge_label=FLAGS.merge_label)
            labels = slim.one_hot_encoding(
                        labels, FLAGS.NUM_CLASSES)
            if hasattr(network_fn, 'rnn_part'):
                assert load_batch_size % FLAGS.n_steps == 0
                total_video_num = int(load_batch_size / FLAGS.n_steps)
                # Split images and labels for cnn
                split_images = tf.split(images, deploy_config.num_clones, 0)
                cnn_labels = labels
                if FLAGS.merge_label:
                    cnn_labels = tf.reshape(cnn_labels, [1, -1, FLAGS.NUM_CLASSES])
                    cnn_labels = tf.tile(cnn_labels, [FLAGS.n_steps, 1, 1])
                    cnn_labels = tf.reshape(cnn_labels, [-1, FLAGS.NUM_CLASSES])
                split_cnn_labels = tf.split(cnn_labels, deploy_config.num_clones, 0)
                # Split labels for rnn
                if not FLAGS.merge_label:
                    split_rnn_labels = tf.reshape(labels, [FLAGS.n_steps, total_video_num, FLAGS.NUM_CLASSES])
                    assert total_video_num % deploy_config.num_clones == 0
                    split_rnn_labels = tf.split(split_rnn_labels, deploy_config.num_clones, 1)
                    each_video_num = int(total_video_num / deploy_config.num_clones)
                    split_rnn_labels = [tf.reshape(label, [FLAGS.n_steps*each_video_num, FLAGS.NUM_CLASSES])
                                        for label in split_rnn_labels]
                else:
                    split_rnn_labels = tf.split(labels, deploy_config.num_clones, 0)
            else:
                batch_queue = slim.prefetch_queue.prefetch_queue(
                        [images, labels], capacity=2 * deploy_config.num_clones)

        ####################
        # Define the model #
        ####################
        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        if hasattr(network_fn, 'rnn_part'):
            cnn_outputs = []
            end_point_outputs = []
            def clone_bn_part(split_batchs, split_cnn_labels, cnn_outputs, end_point_outputs):
                batch = split_batchs[0]
                split_batchs.remove(batch)
                logits, end_points = network_fn(batch)
                cnn_outputs.append(logits)
                end_point_outputs.append(end_points)
                labels = split_cnn_labels[0]
                split_cnn_labels.remove(labels)
                #############################
                # Specify the loss function #
                #############################
                if 'AuxLogits' in end_points:
                    tf.losses.softmax_cross_entropy(
                            logits=end_points['AuxLogits'], onehot_labels=labels,
                            label_smoothing=FLAGS.label_smoothing, weights=0.4, scope='aux_loss')
                return end_points
            def clone_rnn(cnn_outputs, split_rnn_labels, end_point_outputs):
                cnn_output = cnn_outputs[0]
                cnn_outputs.remove(cnn_output)
                end_point_output = end_point_outputs[0]
                end_point_outputs.remove(end_point_output)
                labels = split_rnn_labels[0]
                split_rnn_labels.remove(labels)
                logits, end_points = network_fn.rnn_part(cnn_output)
                end_points.update(end_point_output)
                #############################
                # Specify the loss function #
                #############################
                tf.losses.softmax_cross_entropy(
                            logits=logits, onehot_labels=labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
                return end_points
            # Run BN part, CNN and RNN should have different labels because of the different sample order
            model_deploy.create_clones(deploy_config, clone_bn_part,
                                       [split_images, split_cnn_labels, cnn_outputs, end_point_outputs],
                                       gpu_offset=1)
            # Merge on another GPU to avoid transport data back to original GPUs
            assert len(model_deploy.get_available_gpus()) > deploy_config.num_clones
            with tf.device(deploy_config.clone_device(0)):
                # Concat all clones to one tensor
                cnn_outputs = tf.concat(values=cnn_outputs, axis=0)
                output_shape = cnn_outputs.get_shape().as_list()
                # Reshape to expose the video number dimension
                cnn_outputs = tf.reshape(cnn_outputs, [FLAGS.n_steps, total_video_num]+output_shape[1:])
                # Split in the video number dimension, so that each clone has an input for lstm
                cnn_outputs = tf.split(cnn_outputs, deploy_config.num_clones, 1)
                # Merge n_steps and video number dimension
                cnn_outputs = [tf.reshape(output, [-1]+output_shape[1:]) for output in cnn_outputs]
            # Run RNN part on another GPU #deploy_config.num_clones
            #  clones = model_deploy.create_extra_clones_on_another_gpu(deploy_config, clone_rnn,
                                                #  [cnn_outputs, split_rnn_labels, end_point_outputs])
            clones = model_deploy.create_clones(deploy_config, clone_rnn,
                                                [cnn_outputs, split_rnn_labels, end_point_outputs],
                                                gpu_offset=1)
        else:
            def clone_fn(batch_queue):
                """Allows data parallelism by creating multiple clones of network_fn."""
                images, labels = batch_queue.dequeue()
                logits, end_points = network_fn(images)
                #############################
                # Specify the loss function #
                #############################
                if 'AuxLogits' in end_points:
                    tf.losses.softmax_cross_entropy(
                            logits=end_points['AuxLogits'], onehot_labels=labels,
                            label_smoothing=FLAGS.label_smoothing, weights=0.4, scope='aux_loss')
                tf.losses.softmax_cross_entropy(
                            logits=logits, onehot_labels=labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
                return end_points
            clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                        FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = train_util._configure_learning_rate(global_step)
            optimizer = train_util._configure_optimizer(learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if FLAGS.sync_replicas:
            # If sync_replicas is enabled, the averaging will be done in the chief
            # queue runner.
            optimizer = tf.train.SyncReplicasOptimizer(
                    opt=optimizer,
                    replicas_to_aggregate=FLAGS.replicas_to_aggregate,
                    variable_averages=variable_averages,
                    variables_to_average=moving_average_variables,
                    total_num_replicas=FLAGS.worker_replicas)
        elif FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = train_util._get_variables_to_train()
        # Variables to restore and decay
        variables_to_restore = train_util._get_variables_to_restore()

        #  and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
                    clones,
                    optimizer,
                    var_list=variables_to_train,
                    gate_gradients=optimizer.GATE_OP,
                    colocate_gradients_with_ops=True)

        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Gradient decay and clipping
        if not FLAGS.no_decay:
            # Set up learning rate decay
            lr_mul = {var:0.1 for var in variables_to_restore}
            clones_gradients = tf.contrib.slim.learning.multiply_gradients(
                                        clones_gradients, lr_mul)
        if FLAGS.grad_clipping is not None:
            clones_gradients = tf.contrib.slim.learning.clip_gradient_norms(
                                        clones_gradients, FLAGS.grad_clipping)

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                        name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                        first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')


        ###########################
        # Kicks off the training. #
        ###########################
        sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement)
        slim.learning.train(
                train_tensor,
                logdir=FLAGS.train_dir,
                master=FLAGS.master,
                is_chief=(FLAGS.task == 0),
                init_fn=train_util._get_init_fn(variables_to_restore),
                summary_op=summary_op,
                number_of_steps=FLAGS.max_number_of_steps,
                log_every_n_steps=FLAGS.log_every_n_steps,
                save_summaries_secs=FLAGS.save_summaries_secs,
                save_interval_secs=FLAGS.save_interval_secs,
                sync_optimizer=optimizer if FLAGS.sync_replicas else None,
                trace_every_n_steps=FLAGS.trace_every_n_steps,
                session_config=sess_config)


def test_once(test_data_size, top_k_op, sess, names,
              batch_size_per_gpu, summary_op, summary_writer,
              show_log=False):
    print("Testing......")
    num_eval_batches = int(
            math.ceil(float(test_data_size) / float(batch_size_per_gpu) * float(FLAGS.n_steps)))
    correct = 0
    count = 0
    total = test_data_size
    for i in xrange(num_eval_batches):
        test_start_time = time.time()
        ret, name = sess.run([top_k_op, names])
        correct += np.sum(ret)
        test_duration = time.time() - test_start_time
        count += len(ret)
        cur_accuracy = float(correct)*100/count

        test_examples_per_sec = float(batch_size_per_gpu) / test_duration

        if show_log and i % 100 == 0:
            #  for n in name:
                #  print(n)
            msg = '{:>6.2f}%, {:>6}/{:<6}'.format(cur_accuracy, count, total)
            format_str = ('%s: batch %d, accuracy=%s, (%.1f examples/sec; %.3f '
                    'sec/batch)')
            print (format_str % (datetime.now(), i, msg,
                            test_examples_per_sec, test_duration))
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

    msg = '{:>6.2f}%, {:>6}/{:<6}'.format(cur_accuracy, count, total)
    format_str = ('%s: total batch %d, accuracy=%s, (%.1f examples/sec; %.3f '
            'sec/batch)')
    print (format_str % (datetime.now(), num_eval_batches, msg,
                    test_examples_per_sec, test_duration))
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, num_eval_batches+1)
    summary_writer.flush()


def test():
    # Check training directory.
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Training directory %s not found.", train_dir)
        return

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        ####################
        # Select the network #
        ####################
        network_fn = nets_factory.get_network_fn(
                    FLAGS.model_name,
                    num_classes=FLAGS.NUM_CLASSES,
                    is_training=False)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                    preprocessing_name,
                    is_training=False)

        test_size, test_data, test_label, test_names = async_loader.video_inputs(FLAGS.dataset_list,
                                            FLAGS.dataset_dir, FLAGS.resize_image_size, FLAGS.train_image_size,
                                            FLAGS.batch_size, FLAGS.n_steps, FLAGS.modality, FLAGS.read_stride,
                                            image_preprocessing_fn, shuffle=False,
                                            label_from_one=(FLAGS.labels_offset>0),
                                            length1=FLAGS.length, crop=0,
                                            merge_label=FLAGS.merge_label)
        print("Batch size %d"%test_data.get_shape()[0].value)

        batch_size_per_gpu = FLAGS.batch_size
        global_step_tensor = slim.create_global_step()

        # Calculate the gradients for each model tower.
        logits, end_points = network_fn(test_data)
        if hasattr(network_fn, 'rnn_part'):
            logits, end_points_rnn = network_fn.rnn_part(logits)
            end_points.update(end_points_rnn)
        if not FLAGS.merge_label:
            logits = tf.split(logits, FLAGS.n_steps, 0)[-1]
            test_label = tf.split(test_label, FLAGS.n_steps, 0)[-1]
        top_k_op = tf.nn.in_top_k(logits, test_label, FLAGS.top_k)

        summary_op = tf.summary.merge_all()
        summary_writer =  tf.summary.FileWriter(FLAGS.eval_dir)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step_tensor)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_variables())
            variables_to_restore[global_step_tensor.op.name] = global_step_tensor
        else:
            variables_to_restore = slim.get_variables_to_restore()

        for var in variables_to_restore:
            print("Will restore %s"%(var.op.name))
        saver = tf.train.Saver(variables_to_restore)
        sv = tf.train.Supervisor(graph=g,
                                   logdir=FLAGS.eval_dir,
                                   summary_op=None,
                                   summary_writer=None,
                                   global_step=None,
                                   saver=None)
        g.finalize()

        with sv.managed_session(
                FLAGS.master, start_standard_services=False, config=None) as sess:
            while True:
                start = time.time()
                tf.logging.info("Starting evaluation at " + time.strftime(
                        "%Y-%m-%d-%H:%M:%S", time.localtime()))
                model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
                if not model_path:
                    tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                                    FLAGS.train_dir)
                else:
                    # Load model from checkpoint.
                    tf.logging.info("Loading model from checkpoint: %s", model_path)
                    saver.restore(sess, model_path)
                    global_step = tf.train.global_step(sess, global_step_tensor.name)
                    tf.logging.info("Successfully loaded %s at global step = %d.",
                                    os.path.basename(model_path), global_step)

                    if global_step > 0:
                        # Start the queue runners.
                        sv.start_queue_runners(sess)

                        # Run evaluation on the latest checkpoint.
                        try:
                            test_once(test_size, top_k_op, sess, test_names,
                                    batch_size_per_gpu, summary_op, summary_writer,
                                    show_log=True)
                        except Exception:  # pylint: disable=broad-except
                            tf.logging.error("Evaluation failed.")
                time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
                if time_to_next_eval > 0:
                    time.sleep(time_to_next_eval)


def async_extract():
    # Check training directory.
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.fatal("Training directory %s not found.", train_dir)
        return

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        ####################
        # Select the network #
        ####################
        network_fn = nets_factory.get_network_fn(
                    FLAGS.model_name,
                    num_classes=FLAGS.NUM_CLASSES,
                    is_training=False)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                    preprocessing_name,
                    is_training=False)

        test_size, test_data, test_label, test_names = async_loader.multi_sample_video_inputs(FLAGS.dataset_list,
                                            FLAGS.dataset_dir, FLAGS.batch_size, FLAGS.n_steps,
                                            FLAGS.modality, FLAGS.read_stride,
                                            FLAGS.resize_image_size, FLAGS.train_image_size,
                                            image_preprocessing_fn,
                                            label_from_one=(FLAGS.labels_offset>0),
                                            sample_num=25,
                                            length1=FLAGS.length,
                                            merge_label=FLAGS.merge_label)
        print("Batch size %d"%test_data.get_shape()[0].value)

        batch_size_per_gpu = FLAGS.batch_size
        global_step_tensor = slim.create_global_step()

        # Calculate the gradients for each model tower.
        predicts, end_points = network_fn(test_data)
        if hasattr(network_fn, 'rnn_part'):
            predicts, end_points_rnn = network_fn.rnn_part(predicts)
            end_points.update(end_points_rnn)
        if not FLAGS.merge_label:
            predicts = tf.split(predicts, FLAGS.n_steps, 0)[-1]
            test_label = tf.split(test_label, FLAGS.n_steps, 0)[-1]
        top_k_op = tf.nn.in_top_k(predicts, test_label, FLAGS.top_k)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step_tensor)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_variables())
            variables_to_restore[global_step_tensor.op.name] = global_step_tensor
        else:
            variables_to_restore = slim.get_variables_to_restore()

        for var in variables_to_restore:
            print("Will restore %s"%(var.op.name))
        saver = tf.train.Saver(variables_to_restore)
        sv = tf.train.Supervisor(graph=g,
                                   logdir=FLAGS.eval_dir,
                                   summary_op=None,
                                   summary_writer=None,
                                   global_step=None,
                                   saver=None)
        g.finalize()

        tf.logging.info("Starting evaluation at " + time.strftime(
                "%Y-%m-%d-%H:%M:%S", time.localtime()))
        model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        if not model_path:
            tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                            FLAGS.train_dir)
        else:
            with sv.managed_session(
                    FLAGS.master, start_standard_services=False, config=None) as sess:
                # Load model from checkpoint.
                tf.logging.info("Loading model from checkpoint: %s", model_path)
                saver.restore(sess, model_path)
                global_step = tf.train.global_step(sess, global_step_tensor.name)
                tf.logging.info("Successfully loaded %s at global step = %d.",
                                os.path.basename(model_path), global_step)

                # Start the queue runners.
                sv.start_queue_runners(sess)

                # Run evaluation on the latest checkpoint.
                print("Extracting......")
                num_eval_batches = int(
                        math.ceil(float(test_size) / float(batch_size_per_gpu) * float(FLAGS.n_steps)))
                assert (num_eval_batches*batch_size_per_gpu/FLAGS.n_steps) == test_size
                correct = 0
                count = 0
                for i in xrange(num_eval_batches):
                    test_start_time = time.time()
                    ret, pre, name = sess.run([top_k_op, predicts, test_names])
                    correct += np.sum(ret)
                    for b in xrange(pre.shape[0]):
                        fp = open('%s/%s'%(FLAGS.feature_dir, os.path.basename(name[b])), 'a')
                        for f in xrange(pre.shape[1]):
                            fp.write('%f '%pre[b, f])
                        fp.write('\n')
                        fp.close()
                    test_duration = time.time() - test_start_time
                    count += len(ret)
                    cur_accuracy = float(correct)*100/count

                    test_examples_per_sec = float(batch_size_per_gpu) / test_duration

                    if i % 100 == 0:
                        msg = '{:>6.2f}%, {:>6}/{:<6}'.format(cur_accuracy, count, test_size)
                        format_str = ('%s: batch %d, accuracy=%s, (%.1f examples/sec; %.3f '
                                'sec/batch)')
                        print (format_str % (datetime.now(), i, msg,
                                        test_examples_per_sec, test_duration))

                msg = '{:>6.2f}%, {:>6}/{:<6}'.format(cur_accuracy, count, test_size)
                format_str = ('%s: total batch %d, accuracy=%s, (%.1f examples/sec; %.3f '
                        'sec/batch)')
                print (format_str % (datetime.now(), num_eval_batches, msg,
                                test_examples_per_sec, test_duration))


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'test':
        test()
    else:
        async_extract()


if __name__ == '__main__':
    tf.app.run()
