# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Contains convenience wrappers for typical Neural Network TensorFlow layers.

   Additionally it maintains a collection with update_ops that need to be
   updated after the ops have been computed, for exmaple to update moving means
   and moving variances of batch_norm.

   Ops that have different behavior during training or eval have an is_training
   parameter. Additionally Ops that contain variables.variable have a trainable
   parameter, which control if the ops variables are trainable or not.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl

import numpy as np
from ops import ops

try:
    xrange
except NameError:
    xrange = range
_linear = ops._linear
RNNCell = tf.contrib.rnn.RNNCell
LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple


def shuttleNet(inputs, n_steps, num_mem, mem_dim,
            num_round,
            echocell,
            lstm_type="encell",
            activation=tf.tanh,
            num_frames=None,
            scope=None,
            reuse=None):
    with tf.variable_scope(scope, 'LSTM', [inputs], reuse=reuse):
        if lstm_type == "encell":
            input_shape = inputs.get_shape()
            inputs = tf.reshape(inputs, [input_shape[0].value, -1])
            batch_size = int(input_shape[0].value / n_steps)
            if echocell == "BASIC":
                cell = tf.contrib.rnn.BasicLSTMCell(mem_dim, forget_bias=1.0,
                                                    state_is_tuple=False,
                                                    activation=activation)
            elif echocell == "LSTM":
                cell = tf.contrib.rnn.LSTMCell(mem_dim, state_is_tuple=False,
                                               activation=activation)
            elif echocell == "GRU":
                cell = ops.GRUCell(mem_dim, activation=activation)
            elif echocell == "tfGRU":
                cell = tf.contrib.rnn.GRUCell(mem_dim, activation=activation)
            elif echocell == "GRUBlock":
                cell = tf.contrib.rnn.GRUBlockCell(mem_dim)
            elif echocell == "MEM":
                cell = None
            else:
                raise NotImplementedError("echocell %s is not supported."%(echocell))
            en_cell = ENCell(batch_size, num_mem, num_round, 1,
                             cell=cell,
                             echocell=echocell)
        else:
            raise ValueError("lstm_type %s is not in lstm and conv."%lstm_type)
        initial_state = None

        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        inputs = tf.split(inputs, int(n_steps), int(0)) # n_steps * (batch_size, n_hidden)

        # Get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(en_cell, inputs, initial_state=initial_state,
                                    dtype=tf.float32, sequence_length=num_frames, scope=scope)
        outputs = tf.concat(outputs, 0)
        return outputs


class MemGrid(RNNCell):
    """Memory grid."""
    def __init__(self, batch_size, mem_size, mem_dim, name,
                 activation=tanh, dummy_value=0.0, stddev=0.5):
        self._mem_size = mem_size
        self._mem_dim = mem_dim
        self._activation = activation
        self._batch_size = batch_size

        # memory
        M_init = tf.get_variable("%s_M_init"%name, [1, self._mem_size, self._mem_dim], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        self._memory = tf.tile(M_init, [batch_size, 1, 1], name='%s_Tile_M'%name)

    def dlinear(self, input_, output_size, stddev=0.5,
                        is_range=False, squeeze=False,
                        name=None, reuse=None):
        """Applies a linear transformation to the incoming data.
            Args:
                input: a 2-D or 1-D data (`Tensor` or `ndarray`)
                output_size: the size of output matrix or vector
        """
        with tf.variable_scope("dlinear", reuse=reuse):
            if type(input_) == np.ndarray:
                shape = input_.shape
            else:
                shape = input_.get_shape().as_list()

            is_vector = False
            if len(shape) == 1:
                is_vector = True
                input_ = tf.reshape(input_, [1, -1])
                input_size = shape[0]
            elif len(shape) == 2:
                input_size = shape[1]
            else:
                raise ValueError("Linear expects shape[1] of inputuments: %s" % str(shape))

            w_name = "%s_w" % name if name else None
            b_name = "%s_b" % name if name else None

            w = tf.get_variable(w_name, [input_size, output_size], tf.float32,
                                tf.random_normal_initializer(stddev=stddev))
            mul = tf.matmul(input_, w)

            if is_range:
                def identity_initializer(tensor):
                    def _initializer(shape, dtype=tf.float32):
                        return tf.identity(tensor)
                    return _initializer

                range_ = tf.reverse(tf.range(1, output_size+1, 1), [True])
                b = tf.get_variable(b_name, [output_size], tf.float32,
                                    identity_initializer(tf.cast(range_, tf.float32)))
            else:
                b = tf.get_variable(b_name, [output_size], tf.float32,
                                    tf.random_normal_initializer(stddev=stddev))

            if squeeze:
                output = tf.squeeze(tf.nn.bias_add(mul, b))
            else:
                output = tf.nn.bias_add(mul, b)

            if is_vector:
                return tf.reshape(output, [-1])
            else:
                return output

    @property
    def state_size(self):
        return self._mem_size*self._mem_dim

    @property
    def output_size(self):
        return self._mem_dim

    def unbalance_linear(self, args, output_size, bias, bias_start=0.0, scope=None):
        inputs = args[0]
        memory = args[1]
        with tf.variable_scope(scope or "UnbalanceLinear"):
            oi = _linear(inputs, output_size, False, scope='OI')
            oi = tf.reshape(oi, [self._batch_size, 1, output_size])

            memory = tf.reshape(memory, [self._batch_size * self._mem_size, self._mem_dim])
            os = _linear(memory, output_size, bias, bias_start, scope='OS')
            os = tf.reshape(os, [self._batch_size, self._mem_size, output_size])
            return oi + os

    def __call__(self, inputs, state, scope=None):
        """Memory grid (MemGrid) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):  # "MemGrid"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                r, u = tf.split(self.unbalance_linear([inputs, self._memory],
                                                    2 * self._mem_dim, True, 1.0), 2, 2)
                r, u = sigmoid(r), sigmoid(u)
            with tf.variable_scope("Candidate"):
                c = self._activation(self.unbalance_linear([inputs, r * self._memory],
                                            self._mem_dim, True))
            # Decide which line to write: line weights
            l = att_weight(inputs, tf.concat([c, self._memory], 2), self.echocell, scope="Line_weights")
            l = tf.reshape(l, [self._batch_size, self._mem_size, 1])
            t_memory = u * self._memory + (1 - u) * c
            self._memory = self._memory * (1 - l) + t_memory * l

            #  hl = att_weight(inputs, self._memory, echocell, scope="hidden_lw")
            #  hl = tf.reshape(hl, [self._batch_size, self._mem_size, 1])
            #  output = tf.reduce_sum(hl * self._memory, 1)
            output = tf.reduce_sum(l * self._memory, 1)
            output = tf.reshape(output, [self._batch_size, self._mem_dim])

            return output, state


class ENCell(RNNCell):
    """Echo Network cell."""

    def __init__(self, batch_size, num_mem, num_round, input_offset,
                 cell=None,
                 echocell=None,
                 mem_size=2,
                 mem_dim=1024,
                 activation=tanh,
                 dummy_value=0.0):
        """
        args:
            num_mem: number of cells
            mem_size: number of memory lines, only work for MemGrid
            mem_dim: length of memory line, only work for MemGrid
            num_round:  the round number of processing in the cell
        """
        self._batch_size = batch_size
        self._num_mem = num_mem
        self._mem_dim = mem_dim
        self._num_round = num_round
        self._input_offset = input_offset
        if cell is None:
            self.check = True
            self._mem_cells = [MemGrid(batch_size, mem_size, mem_dim, "Mem_%d"%i,
                                    activation=activation, dummy_value=dummy_value)
                            for i in xrange(num_mem)]
        else:
            self.check = False
            self._mem_cells = [cell] * num_mem
        self.echocell = echocell

    @property
    def state_size(self):
        return self._mem_cells[0].state_size * self._num_mem * self._num_round

    @property
    def output_size(self):
        return self._mem_cells[0].output_size

    def __call__(self, inputs, state, scope=None):
        if self.check and self._num_round > 1 and inputs.get_shape()[1].value != self._mem_dim:
            raise NotImplementedError("When running with a round larger than 1, \
                                      the input dimension[%d] should be equal to \
                                      mem_dim[%d]."%(inputs.get_shape()[1].value, self._mem_dim))
        """ with nunits (EN) cells."""
        states = tf.split(state, self._num_mem * self._num_round, 1, name='state_split')
        with tf.variable_scope(scope or type(self).__name__):  # "ENCell"
            prev_outputs = [inputs] * self._num_mem
            #  final_outputs = []
            with tf.variable_scope("Computing") as varscope:  # "ENCell"
                for r in xrange(self._num_round):
                    final_outputs = []
                    now_outputs = []
                    if r > 0:
                        varscope.reuse_variables()

                    for i in range(self._num_mem):
                        output, state = self._mem_cells[i](prev_outputs[(i+self._input_offset)%self._num_mem],
                                                           states[r*self._num_mem+i],
                                                           scope="Mem_%d"%i)
                        final_outputs.append(output)
                        now_outputs.append(output)
                        states[r*self._num_mem+i] = state
                    prev_outputs = now_outputs

            if len(final_outputs) > 1:
                for i in xrange(len(final_outputs)):
                    final_outputs[i] = tf.reshape(final_outputs[i], [self._batch_size, 1, self.output_size])
                outputs_concat = tf.concat(final_outputs, 1)
                ow = att_weight(inputs, outputs_concat, self.echocell, scope="ow")
                ow = tf.reshape(ow, [self._batch_size, len(final_outputs), 1])
                output = tf.reduce_sum(ow * outputs_concat, 1)
                output = tf.reshape(output, [self._batch_size, self.output_size])
            else:
                output = final_outputs[0]
            state = tf.concat(states, 1)
            return output, state


def att_weight(decoder_inputs, attention_states,
               echocell=None,
               scope=None):
    """
    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x cell.input_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        num_heads: Number of attention heads that read from attention_states.
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".
    """
    if decoder_inputs is None:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                        % attention_states.get_shape())

    with tf.variable_scope(scope or "attention_decoder"):
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = tf.reshape(
            attention_states, [-1, attn_length, 1, attn_size])
        attention_vec_size = attn_size  # Size of query vectors for attention.
        k = tf.get_variable("AttnW",
                                        [1, 1, attn_size, attention_vec_size])
        hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
        v = tf.get_variable("AttnV",
                                            [attention_vec_size])

        """Put attention masks on hidden using hidden_features and decoder_inputs."""
        with tf.variable_scope("Attention"):
            if echocell == 'tfGRU':
                y = core_rnn_cell_impl._linear(decoder_inputs, attention_vec_size, True)
            else:
                y = _linear(decoder_inputs, attention_vec_size, True)
            y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = tf.reduce_sum(
                v * tf.tanh(hidden_features + y), [2, 3])
            #  s = tf.Print(s, [s], summarize=80)
            a = tf.nn.softmax(s)
            # Now calculate the attention-weighted vector d.
            return a


def lstm(inputs, n_steps, num_units,
        is_training=True,
        num_frames=None,
        lstm_type="GRU",
        keep_prob=1.0,
        scope=None):
    with tf.variable_scope(scope, 'LN_LSTM', [inputs]):
        input_shape = inputs.get_shape()
        inputs = tf.reshape(inputs, [input_shape[0].value, -1])
        if lstm_type == "GRU":
            lstm_cell = tf.contrib.rnn.GRUCell(num_units)
        elif lstm_type == "BASIC":
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
        elif lstm_type == "LSTM":
            lstm_cell = tf.contrib.rnn.LSTMCell(num_units)
        else:
            raise ValueError("lstm_type %s is not in lstm and conv."%lstm_type)
        if is_training and keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        initial_state = None

        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        inputs = tf.split(inputs, int(n_steps), int(0)) # n_steps * (batch_size, n_hidden)

        # Get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, inputs, initial_state=initial_state,
                                    dtype=tf.float32, sequence_length=num_frames, scope=scope)
        outputs = tf.concat(outputs, 0)
        return outputs
