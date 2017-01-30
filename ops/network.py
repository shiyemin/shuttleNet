import tensorflow as tf
from tensorflow.python.training import moving_averages

# Used to keep the update ops done by batch_norm.
UPDATE_OPS_COLLECTION = tf.GraphKeys.UPDATE_OPS

DEFAULT_PADDING = 'SAME'
networks_list = []

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        if isinstance(layer_output, list) or isinstance(layer_output, tuple):
            for i in xrange(len(layer_output)):
                self.layers["%s_%d"%(name, i)] = layer_output[i]
        else:
            self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        self.setup()
        #  tf.add_to_collection('MyNetwork', self)
        networks_list.append(self)

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def add_layer(self, layer, name):
        self.layers[name] = layer
        self.feed(layer)
        return self

    def get_output(self, layer_name=''):
        if layer_name:
            try:
                return self.layers[layer_name]
            except KeyError:
                print self.layers.keys()
                raise KeyError('Unknown layer name fed: %s'%layer)
        else:
            return self.terminals[-1]

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer, wd=None, trainable=True, collections=None):
        collections = list(collections or [])
        collections += [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES]
        regularizer = None
        if wd is not None:
            regularizer = tf.contrib.layers.l2_regularizer(wd)
        var = tf.get_variable(name, shape, trainable=self.trainable if trainable else False,
                            initializer=initializer,
                            regularizer=regularizer,
                            collections=collections)
        return var

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name,
             weight_decay=None,
             bias_decay=None,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True,
             initializer=None,
             trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i/group, c_o],
                                   initializer=initializer,
                                   wd=weight_decay,
                                   trainable=trainable)
            if group==1:
                output = convolve(input, kernel)
            else:
                input_groups = tf.split_v(input, group, 3)
                kernel_groups = tf.split_v(kernel, group, 3)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                output = tf.concat_v2(output_groups, 3)
            if biased:
                biases = self.make_var('biases', [c_o],
                                    initializer=tf.constant_initializer(0.0),
                                    wd=bias_decay,
                                    trainable=trainable)
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        var = tf.nn.relu(input, name=name)
        return var

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def dropout(self, input, name, keep_prob=0.5):
        if self.trainable and keep_prob < 1:
            with tf.name_scope(name, 'Dropout', [input]):
                return tf.nn.dropout(input, keep_prob)
        else:
            return input

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat_v2(values=inputs, axis=axis, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def batch_normalization(self, input, name,
                            scale_offset=True,
                            relu=False,
                            decay=0.999,
                            moving_vars='moving_vars'):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name):
            axis = list(range(len(input.get_shape()) - 1))
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape,
                                 initializer=tf.ones_initializer(),
                                 trainable=self.trainable)
                offset = self.make_var('offset', shape=shape,
                                initializer=tf.zeros_initializer,
                                trainable=self.trainable)
            else:
                scale, offset = (None, None)
            # Create moving_mean and moving_variance add them to
            # GraphKeys.MOVING_AVERAGE_VARIABLES collections.
            moving_collections = [moving_vars, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
            moving_mean = self.make_var('mean',
                                            shape,
                                            initializer=tf.zeros_initializer,
                                            trainable=False,
                                            collections=moving_collections)
            moving_variance = self.make_var('variance',
                                                shape,
                                                initializer=tf.ones_initializer(),
                                                trainable=False,
                                                collections=moving_collections)
            if self.trainable:
                # Calculate the moments based on the individual batch.
                mean, variance = tf.nn.moments(input, axis)

                update_moving_mean = moving_averages.assign_moving_average(
                    moving_mean, mean, decay)
                tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay)
                tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
            else:
                # Just use the moving_mean and moving_variance.
                mean = moving_mean
                variance = moving_variance
            output = tf.nn.batch_normalization(
                input,
                mean=mean,
                variance=variance,
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output

    @layer
    def fc(self, input, num_out, name, weight_decay=None, bias_decay=None, relu=True, initializer=None, trainable=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims==4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1].value))
            weights = self.make_var('weights', shape=[dim, num_out],
                                    initializer=initializer,
                                    wd=weight_decay,
                                    trainable=trainable)
            biases = self.make_var('biases', [num_out],
                                   initializer=tf.constant_initializer(0.0),
                                   wd=bias_decay,
                                   trainable=trainable)
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        var = tf.nn.softmax(input, name)
        return var

