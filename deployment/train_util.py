from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from ops.network import networks_list

slim = tf.contrib.slim
tf.app.flags.DEFINE_float('power', 1.0, 'The power parameter for polynomial decay.')
FLAGS = tf.app.flags.FLAGS

def _configure_learning_rate(global_step):
    """Configures the learning rate.

    Args:
        global_step: The global_step tensor.

    Returns:
        A `Tensor` representing the learning rate.

    Raises:
        ValueError: if
    """
    decay_steps = eval(FLAGS.decay_iteration)

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True,
                                        name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.end_learning_rate,
                                        power=FLAGS.power,
                                        cycle=False,
                                        name='polynomial_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'piecewise':
        if isinstance(decay_steps, int):
            decay_steps = [decay_steps*i for i in xrange(1, 6)]
        decay_steps = [np.int64(i) for i in decay_steps]
        lr_values = [FLAGS.learning_rate*FLAGS.learning_rate_decay_factor**i for i in xrange(len(decay_steps)+1)]
        return tf.train.piecewise_constant(
                                        global_step,
                                        decay_steps,
                                        lr_values)
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                        FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
        learning_rate: A scalar or `Tensor` learning rate.

    Returns:
        An instance of an optimizer.

    Raises:
        ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate,
                    rho=FLAGS.adadelta_rho,
                    epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
                    learning_rate,
                    initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
                    learning_rate,
                    beta1=FLAGS.adam_beta1,
                    beta2=FLAGS.adam_beta2,
                    epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
                    learning_rate,
                    learning_rate_power=FLAGS.ftrl_learning_rate_power,
                    initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
                    l1_regularization_strength=FLAGS.ftrl_l1,
                    l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
                    learning_rate,
                    momentum=FLAGS.momentum,
                    name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
                    learning_rate,
                    decay=FLAGS.rmsprop_decay,
                    momentum=FLAGS.rmsprop_momentum,
                    epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _add_variables_summaries(learning_rate):
    summaries = []
    for variable in slim.get_model_variables():
        summaries.append(tf.summary.histogram(variable.op.name, variable))
    summaries.append(tf.summary.scalar('training/Learning Rate', learning_rate))
    return summaries


def _load_weights():
    data_dict = np.load(FLAGS.npy_weights).item()
    networks = networks_list
    assert len(networks) == FLAGS.num_clones
    network = networks[0]
    ops = []
    for key in data_dict:
        try:
            if key in network.layers.keys():
                print('Restoring layer %s'%key)
                with tf.variable_scope(key, reuse=True):
                    for subkey, data in data_dict[key].iteritems():
                        var = tf.get_variable(subkey)
                        if (len(data.shape) == 4 and
                            data.shape[2] == 3 and
                            var.get_shape()[2].value > 3):
                            print("Modifying %s of %s: %s -> %s"%(subkey, key,
                                    str(data.shape), str(var.get_shape().as_list())))
                            new_data = np.zeros(var.get_shape().as_list())
                            new_data[:,:,:,:] = np.mean(data, axis=2, keepdims=True)
                            data = new_data
                        ops.append(var.assign(data))
            if "copy_%s"%key in network.layers.keys():
                print('Restoring layer copy_%s'%key)
                with tf.variable_scope("copy_%s"%key, reuse=True):
                    for subkey, data in data_dict[key].iteritems():
                        var = tf.get_variable(subkey)
                        if (len(data.shape) == 4 and
                            data.shape[2] == 3 and
                            var.get_shape()[2].value > 3):
                            print("Modifying %s of %s: %s -> %s"%(subkey, key,
                                    str(data.shape), str(var.get_shape().as_list())))
                            new_data = np.zeros(var.get_shape().as_list())
                            new_data[:,:,:,:] = np.mean(data, axis=2, keepdims=True)
                            data = new_data
                        ops.append(var.assign(data))
        except ValueError:
            if not FLAGS.ignore_missing_vars:
                raise
    return tf.group(*ops)


def _get_init_fn(variables_to_restore):
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
        An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None and FLAGS.npy_weights is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
                'Ignoring --checkpoint_path because a checkpoint already exists in %s'
                % FLAGS.train_dir)
        return None

    if FLAGS.npy_weights is not None:
        load_op = _load_weights()
        tf.logging.info('Fine-tuning from %s' % FLAGS.npy_weights)
        def init_with_weight(sess):
            sess.run(load_op)
        return init_with_weight

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    if FLAGS.no_restore_exclude:
        variables_to_restore = slim.get_model_variables()
    return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_restore():
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                    for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            if FLAGS.no_decay:
                print('Will restore %s'%(var.op.name))
            elif FLAGS.no_restore_exclude:
                print('Will decay the lr of %s'%(var.op.name))
            else:
                print('Will restore and decay the lr of %s'%(var.op.name))
            variables_to_restore.append(var)
        else:
            print('Will not decay and restore the lr of %s'%(var.op.name))
    return variables_to_restore


def _get_variables_to_train():
    """Returns a list of variables to train.

    Returns:
        A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

