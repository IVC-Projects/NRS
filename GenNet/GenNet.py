"""Implements a voxel flow model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#import tensorflow.contrib.slim as slim
import tf_slim as slim
from utils.loss_utils import l1_loss, l2_loss, vae_loss
from utils.geo_layer_utils import vae_gaussian_layer
from utils.geo_layer_utils import bilinear_interp
from utils.geo_layer_utils import meshgrid

FLAGS = tf.app.flags.FLAGS
epsilon = 0.001


class Voxel_flow_model(object):
    def __init__(self, is_train=True, is_extrapolation=False):
        self.is_train = is_train
        self.is_extrapolation = is_extrapolation

    def inference(self, input_images):
        """Inference on a set of input_images.
        Args:
        """
        return self._build_model(input_images)

    def total_var(self, images):
        pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
        pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
        tot_var = (tf.reduce_mean(tf.sqrt(tf.square(pixel_dif1) + epsilon**2)) + tf.reduce_mean(tf.sqrt(tf.square(pixel_dif2) + epsilon**2)))
        return tot_var

    def loss(self, predictions, targets):
        """Compute the necessary loss for training.
        Args:
        Returns:
        """
        # self.reproduction_loss = l1_loss(predictions, targets)
        
        self.reproduction_loss = tf.reduce_mean(tf.sqrt(tf.square(predictions - targets) + epsilon**2))

        self.motion_loss = self.total_var(self.flow) # 运动矢量的loss
        self.mask_loss = self.total_var(self.mask)   # 遮罩层的loss

        # return [self.reproduction_loss, self.prior_loss]
        return self.reproduction_loss + 0.01 * self.motion_loss + 0.005 * self.mask_loss

    def l1loss(self, predictions, targets):
        self.reproduction_loss = l1_loss(predictions, targets)
        return self.reproduction_loss

    def resblock(self, temp_tensor, convId, times):
        skip_tensor = temp_tensor

        for i in range(times):
            temp_tensor = slim.conv2d(temp_tensor, 3, [3, 3], stride=1, scope='res_conv%02d'%convId,reuse=tf.AUTO_REUSE)
        # skip + out_tensor
        out_tensor = tf.add(skip_tensor, temp_tensor)

        return out_tensor


    def _build_model(self, input_images):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            # Define network
            batch_norm_params = {
                'decay': 0.9997,
                'epsilon': 0.001,
                'is_training': self.is_train,
            }
            with slim.arg_scope([slim.batch_norm], is_training=self.is_train, updates_collections=None):
                with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                    normalizer_params=batch_norm_params):

                    x1 = slim.conv2d(input_images, 32, [5, 5], stride=1, scope='conv1')

                    net = slim.max_pool2d(x1, [2, 2], scope='pool1')
                    x2 = slim.conv2d(net, 64, [3, 3], stride=1, scope='conv2')

                    net = slim.max_pool2d(x2, [2, 2], scope='pool2')
                    x3 = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv3')

                    net = slim.max_pool2d(x3, [2, 2], scope='pool3')
                    x4 = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv4')

                    net = slim.max_pool2d(x4, [2, 2], scope='pool4')
                    net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv5')

                    net = tf.image.resize_bilinear(net, [x4.get_shape().as_list()[1], x4.get_shape().as_list()[2]])
                    net = slim.conv2d(tf.concat([net, x4], -1), 256, [3, 3], stride=1, scope='conv6')

                    net = tf.image.resize_bilinear(net, [x3.get_shape().as_list()[1], x3.get_shape().as_list()[2]])
                    net = slim.conv2d(tf.concat([net, x3], -1), 128, [3, 3], stride=1, scope='conv7')

                    net = tf.image.resize_bilinear(net, [x2.get_shape().as_list()[1], x2.get_shape().as_list()[2]])
                    net = slim.conv2d(tf.concat([net, x2], -1), 64, [3, 3], stride=1, scope='conv8')

                    net = tf.image.resize_bilinear(net, [x1.get_shape().as_list()[1], x1.get_shape().as_list()[2]])
                    y0 = slim.conv2d(tf.concat([net, x1], -1), 32, [3, 3], stride=1, scope='conv9')

        net = slim.conv2d(y0, 3, [5, 5], stride=1, activation_fn=tf.tanh,
                          normalizer_fn=None, scope='conv10')
        net_copy = net

        flow = net[:, :, :, 0:2]
        mask = tf.expand_dims(net[:, :, :, 2], 3)
        # print("***********************************************************************")
        # print(net.shape)
        # print(flow.shape)
        # print(net[:, :, :, 0])
        # print(net[:, :, :, 1])
        # print(net[:, :, :, 2])
        # print(mask.shape)
        # print(x.shape)
        # print(x.get_shape().as_list()[1])
        # print("***********************************************************************")
        self.flow = flow


        grid_x, grid_y = meshgrid(x1.get_shape().as_list()[1], x1.get_shape().as_list()[2])
        # print("this is grid_x:",grid_x.shape)
        # print(grid_y.shape)
        grid_x = tf.tile(grid_x, [FLAGS.batch_size, 1, 1])
        grid_y = tf.tile(grid_y, [FLAGS.batch_size, 1, 1])

        # print(grid_x.shape)
        flow = 0.5 * flow

        flow_ratio = tf.constant([255.0 / (x1.get_shape().as_list()[2]-1), 255.0 / (x1.get_shape().as_list()[1]-1)])
        flow = flow * tf.expand_dims(tf.expand_dims(tf.expand_dims(flow_ratio, 0), 0), 0)

        if self.is_extrapolation:
            coor_x_1 = grid_x + flow[:, :, :, 0] * 2
            coor_y_1 = grid_y + flow[:, :, :, 1] * 2
            coor_x_2 = grid_x + flow[:, :, :, 0]
            coor_y_2 = grid_y + flow[:, :, :, 1]
        else:
            coor_x_1 = grid_x + flow[:, :, :, 0]
            coor_y_1 = grid_y + flow[:, :, :, 1]
            coor_x_2 = grid_x - flow[:, :, :, 0]
            coor_y_2 = grid_y - flow[:, :, :, 1]

        output_1 = bilinear_interp(input_images[:, :, :, 0:3], coor_x_1, coor_y_1, 'interpolate')
        output_2 = bilinear_interp(input_images[:, :, :, 3:6], coor_x_2, coor_y_2, 'interpolate')

        self.warped_img1 = output_1
        self.warped_img2 = output_2

        self.warped_flow1 = bilinear_interp(-flow[:, :, :, 0:3]*0.5, coor_x_1, coor_y_1, 'interpolate')
        self.warped_flow2 = bilinear_interp(flow[:, :, :, 0:3]*0.5, coor_x_2, coor_y_2, 'interpolate')

        mask = 0.5 * (1.0 + mask)
        self.mask = mask
        mask = tf.tile(mask, [1, 1, 1, 3])
        net = tf.multiply(mask, output_1) + tf.multiply(1.0 - mask, output_2)

        for i in range(8):
            net = self.resblock(net, i, 4)

        return [net, net_copy]


