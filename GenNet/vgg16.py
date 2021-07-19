# Adapted from : VGG 16 model : https://github.com/machrisaa/tensorflow-vgg
import time
import os
import inspect

import numpy as np
from termcolor import colored
import tensorflow as tf

from losses import sigmoid_cross_entropy_balanced
import pdb
#from io import IO

VGG_MEAN = [103.939, 116.779, 123.68]
class Vgg16():

    def __init__(self, input_image,reuse=None):

        # self.cfgs 1= cfgs
#        self.io = IO()

        base_path = os.path.abspath(os.path.dirname(__file__))
        print(base_path)
        weights_file = os.path.join(base_path, 'vgg16.npy')
        print(weights_file)

        self.data_dict = np.load(weights_file, allow_pickle=True, encoding='latin1').item()
        # self.io.print_info("Model weights loaded from {}".format(self.cfgs['model_weights_path']))
        print((input_image+tf.ones_like(input_image)))
        rgb_scaled = tf.subtract((input_image+tf.ones_like(input_image)),2)*255.
        red, green, blue = tf.split(rgb_scaled, 3, 3)

        self.images = tf.concat([blue - VGG_MEAN[0],
                        green - VGG_MEAN[1],
                        red - VGG_MEAN[2]],
                        3)
        # self.images = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], self.cfgs[run]['n_channels']])
        # self.edgemaps = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], 1])
        self.define_model(reuse=reuse)

    def define_model(self,reuse=None):

        """
        Load VGG params from disk without FC layers A
        Add branch layers (with deconv) after each CONV block
        """
        with tf.compat.v1.variable_scope('hed'):
            start_time = time.time()
            self.conv1_1 = self.conv_layer_vgg(self.images, "conv1_1")
            self.conv1_2 = self.conv_layer_vgg(self.conv1_1, "conv1_2")
            self.side_1 = self.side_layer(self.conv1_2, "side_1", 1,reuse=reuse)
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            # self.io.print_info('Added CONV-BLOCK-1+SIDE-1')

            self.conv2_1 = self.conv_layer_vgg(self.pool1, "conv2_1")
            self.conv2_2 = self.conv_layer_vgg(self.conv2_1, "conv2_2")
            self.side_2 = self.side_layer(self.conv2_2, "side_2", 2,reuse=reuse)
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            # self.io.print_info('Added CONV-BLOCK-2+SIDE-2')

            self.conv3_1 = self.conv_layer_vgg(self.pool2, "conv3_1")
            self.conv3_2 = self.conv_layer_vgg(self.conv3_1, "conv3_2")
            self.conv3_3 = self.conv_layer_vgg(self.conv3_2, "conv3_3")
            self.side_3 = self.side_layer(self.conv3_3, "side_3", 4,reuse=reuse)
            self.pool3 = self.max_pool(self.conv3_3, 'pool3')

            # self.io.print_info('Added CONV-BLOCK-3+SIDE-3')

            self.conv4_1 = self.conv_layer_vgg(self.pool3, "conv4_1")
            self.conv4_2 = self.conv_layer_vgg(self.conv4_1, "conv4_2")
            self.conv4_3 = self.conv_layer_vgg(self.conv4_2, "conv4_3")
            self.side_4 = self.side_layer(self.conv4_3, "side_4", 8,reuse=reuse)
            self.pool4 = self.max_pool(self.conv4_3, 'pool4')

            # self.io.print_info('Added CONV-BLOCK-4+SIDE-4')

            self.conv5_1 = self.conv_layer_vgg(self.pool4, "conv5_1")
            self.conv5_2 = self.conv_layer_vgg(self.conv5_1, "conv5_2")
            self.conv5_3 = self.conv_layer_vgg(self.conv5_2, "conv5_3")
            self.side_5 = self.side_layer(self.conv5_3, "side_5", 16,reuse=reuse)

            # self.io.print_info('Added CONV-BLOCK-5+SIDE-5')

            self.side_outputs = [self.side_1, self.side_2, self.side_3, self.side_4, self.side_5]
            w_shape = [1, 1, len(self.side_outputs), 1]
            if reuse == True:
                tf.compat.v1.get_variable_scope().reuse_variables()
            self.fuse = self.conv_layer(tf.concat(self.side_outputs, axis=3),
                                    w_shape, name='fuse_1', use_bias=False,
                                    w_init=tf.compat.v1.constant_initializer(0.2))
            #tf.get_variable_scope().reuse == False

            # self.io.print_info('Added FUSE layer')

            # complete output maps from side layer and fuse layers
            self.outputs = self.side_outputs + [self.fuse]

            self.data_dict = None
            # self.io.print_info("Build model finished: {:.4f}s".format(time.time() - start_time))

    def max_pool(self, bottom, name):
        return tf.nn.max_pool2d(input=bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer_vgg(self, bottom, name):
        """
            Adding a conv layer + weight parameters from a dict
        """
        with tf.compat.v1.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(input=bottom, filters=filt, strides=[1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def conv_layer(self, x, W_shape, b_shape=None, name=None,
                   padding='SAME', use_bias=True, w_init=None, b_init=None):

        W = self.weight_variable(W_shape, w_init, 'Variable')
        tf.compat.v1.summary.histogram('weights_{}'.format(name), W)

        if use_bias:
            b = self.bias_variable([b_shape], b_init, 'Variable_1')
            tf.compat.v1.summary.histogram('biases_{}'.format(name), b)

        conv = tf.nn.conv2d(input=x, filters=W, strides=[1, 1, 1, 1], padding=padding)

        return conv + b if use_bias else conv

    def deconv_layer(self, x, upscale, name, padding='SAME', w_init=None):

        x_shape = tf.shape(input=x)
        in_shape = x.shape.as_list()

        w_shape = [upscale * 2, upscale * 2, in_shape[-1], 1]
        strides = [1, upscale, upscale, 1]

        W = self.weight_variable(w_shape, w_init, 'Variable_2')
        tf.compat.v1.summary.histogram('weights_{}'.format(name), W)

        out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], w_shape[2]]) * tf.constant(strides, tf.int32)
        deconv = tf.nn.conv2d_transpose(x, W, out_shape, strides=strides, padding=padding)

        return deconv

    def side_layer(self, inputs, name, upscale,reuse=None):
        """
            https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/examples/hed/train_val.prototxt#L122
            1x1 conv followed with Deconvoltion layer to upscale the size of input image sans color
        """
        with tf.compat.v1.variable_scope(name,reuse=reuse):

            in_shape = inputs.shape.as_list()
            w_shape = [1, 1, in_shape[-1], 1]

            classifier = self.conv_layer(inputs, w_shape, b_shape=1,
                                         w_init=tf.compat.v1.constant_initializer(),
                                         b_init=tf.compat.v1.constant_initializer(),
                                         name=name + '_reduction')

            classifier = self.deconv_layer(classifier, upscale=upscale,
                                           name='{}_deconv_{}'.format(name, upscale),
                                           w_init=tf.compat.v1.truncated_normal_initializer(stddev=0.1))

            return classifier

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def weight_variable(self, shape, initial, name):

        return tf.compat.v1.get_variable(name, shape=shape, initializer=initial)

    def bias_variable(self, shape, initial, name):

        return tf.compat.v1.get_variable(name, shape=shape, initializer=initial)

    def setup_testing(self, session):

        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs for predictions
        """

        self.predictions = []

        for idx, b in enumerate(self.outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            self.predictions.append(output)

    def setup_training(self, session):

        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs
            Compute total loss := side_layer_loss + fuse_layer_loss
            Compute predicted edge maps from fuse layer as pseudo performance metric to track
        """

        self.predictions = []
        self.loss = 0

        self.io.print_warning('Deep supervision application set to {}'.format(self.cfgs['deep_supervision']))

        for idx, b in enumerate(self.side_outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            cost = sigmoid_cross_entropy_balanced(b, self.edgemaps, name='cross_entropy{}'.format(idx))

            self.predictions.append(output)
            if self.cfgs['deep_supervision']:
                self.loss += (self.cfgs['loss_weights'] * cost)

        fuse_output = tf.nn.sigmoid(self.fuse, name='fuse')
        fuse_cost = sigmoid_cross_entropy_balanced(self.fuse, self.edgemaps, name='cross_entropy_fuse')

        self.predictions.append(fuse_output)
        self.loss += (self.cfgs['loss_weights'] * fuse_cost)

        pred = tf.cast(tf.greater(fuse_output, 0.5), tf.int32, name='predictions')
        error = tf.cast(tf.not_equal(pred, tf.cast(self.edgemaps, tf.int32)), tf.float32)
        self.error = tf.reduce_mean(input_tensor=error, name='pixel_error')

        tf.compat.v1.summary.scalar('loss', self.loss)
        tf.compat.v1.summary.scalar('error', self.error)

        self.merged_summary = tf.compat.v1.summary.merge_all()

        self.train_writer = tf.compat.v1.summary.FileWriter(self.cfgs['save_dir'] + '/train', session.graph)
        self.val_writer = tf.compat.v1.summary.FileWriter(self.cfgs['save_dir'] + '/val')
