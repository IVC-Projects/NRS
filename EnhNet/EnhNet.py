import tensorflow as tf
import numpy as np


def resblock(temp_tensor, convId, times=1):
    skip_tensor = temp_tensor
    # ---------------------------------------------------------------------------------------------------------------------
    # Conv, 1x1, filters=192 ,+ ReLU
    conv_w1 = tf.get_variable("conv_%02d_w1" % (convId), [1, 1, 40, 256],
                              initializer=tf.contrib.layers.xavier_initializer())
    conv_b1 = tf.get_variable("conv_%02d_b1" % (convId), [256],
                              initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w1))
    # ---------------------------------------------------------------------------------------------------------------------

    # Conv, 1x1, filters=25
    conv_w2 = tf.get_variable("conv_%02d_w2" % (convId), [1, 1, 256, 24],
                              initializer=tf.contrib.layers.xavier_initializer())
    conv_b2 = tf.get_variable("conv_%02d_b2" % (convId), [24], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w2))
    # ---------------------------------------------------------------------------------------------------------------------

    # Conv, 3x3, filters=32
    conv_w3 = tf.get_variable("conv_%02d_w3" % (convId), [3, 3, 24, 40],
                              initializer=tf.contrib.layers.xavier_initializer())
    conv_b3 = tf.get_variable("conv_%02d_b3" % (convId), [40], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w3))
    # ---------------------------------------------------------------------------------------------------------------------
    for i in range(times):
        temp_tensor = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w1, strides=[1, 1, 1, 1], padding='SAME'), conv_b1))
        temp_tensor = tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w2, strides=[1, 1, 1, 1], padding='SAME'), conv_b2)
        temp_tensor = tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w3, strides=[1, 1, 1, 1], padding='SAME'), conv_b3)

    # skip + out_tensor
    out_tensor = tf.add(skip_tensor, temp_tensor)
    return out_tensor


def model(input_tensor):
    tensor = None
    convId = 0

    conv_00_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 1, 40],
                                initializer=tf.contrib.layers.xavier_initializer())
    conv_00_b = tf.get_variable("conv_%02d_b" % (convId), [40], initializer=tf.constant_initializer(0))
    tensor = tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1, 1, 1, 1], padding='SAME'), conv_00_b)
    convId += 1


    for i in range(8):
        tensor = resblock(tensor, convId, times=2)
        convId += 1

    conv_11_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 40, 1],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_11_b = tf.get_variable("conv_%02d_b" % (convId), [1], initializer=tf.constant_initializer(0))
    tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_11_w, strides=[1, 1, 1, 1], padding='SAME'), conv_11_b)

    tensor = tf.add(tensor, input_tensor)

    return tensor
