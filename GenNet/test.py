"""Train a voxel flow model on ucf101 dataset."""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import scipy.misc
from datetime import datetime
from GenNet import Voxel_flow_model
import scipy as sp
import cv2, re, gc
from vgg16 import Vgg16
from psnr import printPsnr
from UTILS import *

FLAGS = tf.app.flags.FLAGS
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Define necessary FLAGS
# tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', None,
#                            """If specified, restore this pretrained model """
#                            """before beginning any training.""")
tf.app.flags.DEFINE_integer('batch_size', 1, 'The number of samples in each batch.')


# tf.app.flags.DEFINE_string('first', '',
#                            """first image """)
# tf.app.flags.DEFINE_string('second', '',
#                            """second image """)
# tf.app.flags.DEFINE_string('out', '',
#                            """output image """)


def imread(filename):
    """Read image from file.
    Args:
    filename: .
    Returns:
    im_array: .
    """
    im = sp.misc.imread(filename, mode='RGB')
    return im / 127.5 - 1.0


def readyuv(filename, width, height):
    with open(filename, 'rb') as fp:
        Yt = np.zeros(shape=(height, width), dtype='uint8', order='C')
        Ut = np.zeros(shape=(height // 2, width // 2), dtype='uint8', order='C')
        Vt = np.zeros(shape=(height // 2, width // 2), dtype='uint8', order='C')
        for m in range(height):
            for n in range(width):
                Yt[m, n] = int.from_bytes(fp.read(1), byteorder='little', signed=False)
        for m in range(height // 2):
            for n in range(width // 2):
                Ut[m, n] = int.from_bytes(fp.read(1), byteorder='little', signed=False)
        for m in range(height // 2):
            for n in range(width // 2):
                Vt[m, n] = int.from_bytes(fp.read(1), byteorder='little', signed=False)

        img = np.concatenate((Yt.reshape(-1), Ut.reshape(-1), Vt.reshape(-1)))
        print(np.shape(img))
        img = img.reshape((height * 3 // 2, width)).astype('uint8')
        print(np.shape(img))
        RGB_img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB_I420)
        # cv2.imshow("111", RGB_img)
        # cv2.waitKey()

    fp.close()
    del Yt, Ut, Vt, img
    gc.collect()
    return RGB_img / 127.5 - 1.0


def getWH(yuvfileName):  # Train
    print(yuvfileName)
    deyuv = re.compile(r'(.+?)\.')
    deyuvFilename = deyuv.findall(yuvfileName)[0]  # 去yuv后缀的文件名
    print(deyuvFilename)
    if os.path.basename(deyuvFilename).split("_")[0].isdigit():
        wxh = os.path.basename(deyuvFilename).split('_')[3]
    else:
        wxh = os.path.basename(deyuvFilename).split('_')[3]
    w, h = wxh.split('x')
    # print(deyuvFilename,w,h)
    return int(w), int(h)


def cut_extra(video_frame, restore_path, height, width):
    img = cv2.imread(video_frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    img = img[int((img.shape[0] - height) / 2):height + int((img.shape[0] - height) / 2),
          int((img.shape[1] - width) / 2):width + int((img.shape[1] - width) / 2), :]
    cv2.VideoWriter(restore_path,
                    cv2.VideoWriter_fourcc('I', '4', '2', '0'), 24, (width, height)).write(
        np.round(img).astype(np.uint8)[:, :, ::-1])
    os.remove(video_frame)


def load_file_list(directory):
    list = []
    # for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory,y))]:
    #     print(filename)
    #     if filename.split(".")[-1]=="yuv":
    #         list.append(os.path.join(directory,filename))
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_name = os.path.join(root, file)
            if file_name.split("\\")[-1].split("-")[0] == "model.ckpt":
                list.append(file_name)
    return sorted(list)


def test(first, second, out):
    width, height = getWH(first)

    data_frame1 = np.expand_dims(readyuv(first, width, height), 0)
    data_frame3 = np.expand_dims(readyuv(second, width, height), 0)


    ckpts = load_file_list(ckptsPath)
    psnrlist = []

    for i in range(0, len(ckpts), 3):
        ckpt_file_name = "model." + ckpts[i].split(".")[1]

        print(data_frame1.shape)
        if data_frame1.shape[2] >= 2560:

            rec_list = []
            img1_list = patching(data_frame1, 2560, 1600)
            img3_list = patching(data_frame3, 2560, 1600)

            for each in range(len(img1_list)):
                H = img1_list[each].shape[1]
                W = img1_list[each].shape[2]

                adatptive_H = int(np.ceil(H / 32.0) * 32.0)
                adatptive_W = int(np.ceil(W / 32.0) * 32.0)

                pad_up = int(np.ceil((adatptive_H - H) / 2.0))
                pad_bot = int(np.floor((adatptive_H - H) / 2.0))
                pad_left = int(np.ceil((adatptive_W - W) / 2.0))
                pad_right = int(np.floor((adatptive_W - W) / 2.0))
                """Perform test on a trained model."""
                with tf.Graph().as_default():
                    # Create input and target placeholder.

                    input_placeholder = tf.placeholder(tf.float32, shape=(None, H, W, 6))

                    input_pad = tf.pad(input_placeholder,
                                       [[0, 0], [pad_up, pad_bot], [pad_left, pad_right], [0, 0]], 'SYMMETRIC')
                    print(input_pad)
                    print(input_pad[:, :, :, :3])
                    print(input_pad[:, :, :, 3:6])

                    edge_vgg_1 = Vgg16(input_pad[:, :, :, :3], reuse=None)
                    edge_vgg_3 = Vgg16(input_pad[:, :, :, 3:6], reuse=True)
                    print(edge_vgg_1, edge_vgg_3)
                    edge_1 = tf.nn.sigmoid(edge_vgg_1.fuse)
                    edge_3 = tf.nn.sigmoid(edge_vgg_3.fuse)
                    print(edge_vgg_1, edge_vgg_3)
                    edge_1 = tf.reshape(edge_1,
                                        [-1, input_pad.get_shape().as_list()[1], input_pad.get_shape().as_list()[2],
                                         1])
                    edge_3 = tf.reshape(edge_3,
                                        [-1, input_pad.get_shape().as_list()[1], input_pad.get_shape().as_list()[2],
                                         1])
                    print(edge_vgg_1, edge_vgg_3)
                    with tf.variable_scope("Cycle_DVF"):
                        # Prepare model.
                        model = Voxel_flow_model(is_train=False)
                        prediction = model.inference(tf.concat([input_pad, edge_1, edge_3], 3))[0]
                    sess = tf.Session()
                    restorer = tf.train.Saver()
                    restorer.restore(sess,
                                     os.path.join(ckptsPath, ckpt_file_name))

                    feed_dict = {input_placeholder: np.concatenate((img1_list[each], img3_list[each]), 3)}
                    # Run single step update.
                    prediction_np = sess.run(prediction, feed_dict=feed_dict)
                    rec_list.append(prediction_np)

            rec = np.ones(( 1,1600, 2560, 3))
            output = combination2560(rec, rec_list)

        elif data_frame1.shape[2] >= 1920:
            rec_list = []
            img1_list = patching(data_frame1.reshape(1080, 1920), 1920, 1080)
            img3_list = patching(data_frame3.reshape(1080, 1920), 1920, 1080)

            for each in range(len(img1_list)):
                H = img1_list[each].shape[1]
                W = img1_list[each].shape[2]

                adatptive_H = int(np.ceil(H / 32.0) * 32.0)
                adatptive_W = int(np.ceil(W / 32.0) * 32.0)

                pad_up = int(np.ceil((adatptive_H - H) / 2.0))
                pad_bot = int(np.floor((adatptive_H - H) / 2.0))
                pad_left = int(np.ceil((adatptive_W - W) / 2.0))
                pad_right = int(np.floor((adatptive_W - W) / 2.0))
                """Perform test on a trained model."""
                with tf.Graph().as_default():
                    # Create input and target placeholder.

                    input_placeholder = tf.placeholder(tf.float32, shape=(None, H, W, 6))

                    input_pad = tf.pad(input_placeholder,
                                       [[0, 0], [pad_up, pad_bot], [pad_left, pad_right], [0, 0]], 'SYMMETRIC')
                    print(input_pad)
                    print(input_pad[:, :, :, :3])
                    print(input_pad[:, :, :, 3:6])

                    edge_vgg_1 = Vgg16(input_pad[:, :, :, :3], reuse=None)
                    edge_vgg_3 = Vgg16(input_pad[:, :, :, 3:6], reuse=True)
                    print(edge_vgg_1, edge_vgg_3)
                    edge_1 = tf.nn.sigmoid(edge_vgg_1.fuse)
                    edge_3 = tf.nn.sigmoid(edge_vgg_3.fuse)
                    print(edge_vgg_1, edge_vgg_3)
                    edge_1 = tf.reshape(edge_1,
                                        [-1, input_pad.get_shape().as_list()[1], input_pad.get_shape().as_list()[2],
                                         1])
                    edge_3 = tf.reshape(edge_3,
                                        [-1, input_pad.get_shape().as_list()[1], input_pad.get_shape().as_list()[2],
                                         1])
                    print(edge_vgg_1, edge_vgg_3)
                    with tf.variable_scope("Cycle_DVF"):
                        # Prepare model.
                        model = Voxel_flow_model(is_train=False)
                        prediction = model.inference(tf.concat([input_pad, edge_1, edge_3], 3))[0]
                    sess = tf.Session()
                    restorer = tf.train.Saver()
                    restorer.restore(sess,
                                     os.path.join(ckptsPath, ckpt_file_name))

                    feed_dict = {input_placeholder: np.concatenate((img1_list[each], img3_list[each]), 3)}
                    # Run single step update.
                    prediction_np = sess.run(prediction, feed_dict=feed_dict)

                    output = prediction_np[-1, pad_up:adatptive_H - pad_bot, pad_left:adatptive_W - pad_right, :]
                    output = np.round(((output + 1.0) * 255.0 / 2.0)).astype(np.uint8)
                    output = np.dstack((output[:, :, 2], output[:, :, 1], output[:, :, 0]))

                    rec_list.append(output)
            rec = np.ones((1, 1080, 1920, 3))
            output = combination1920(rec, rec_list)

        else:
            H = data_frame1.shape[1]
            W = data_frame1.shape[2]

            adatptive_H = int(np.ceil(H / 32.0) * 32.0)
            adatptive_W = int(np.ceil(W / 32.0) * 32.0)

            pad_up = int(np.ceil((adatptive_H - H) / 2.0))
            pad_bot = int(np.floor((adatptive_H - H) / 2.0))
            pad_left = int(np.ceil((adatptive_W - W) / 2.0))
            pad_right = int(np.floor((adatptive_W - W) / 2.0))
            """Perform test on a trained model."""
            with tf.Graph().as_default():
                # Create input and target placeholder.

                input_placeholder = tf.placeholder(tf.float32, shape=(None, H, W, 6))

                input_pad = tf.pad(input_placeholder,
                                   [[0, 0], [pad_up, pad_bot], [pad_left, pad_right], [0, 0]], 'SYMMETRIC')
                print(input_pad)
                print(input_pad[:, :, :, :3])
                print(input_pad[:, :, :, 3:6])

                edge_vgg_1 = Vgg16(input_pad[:, :, :, :3], reuse=None)
                edge_vgg_3 = Vgg16(input_pad[:, :, :, 3:6], reuse=True)
                print(edge_vgg_1, edge_vgg_3)
                edge_1 = tf.nn.sigmoid(edge_vgg_1.fuse)
                edge_3 = tf.nn.sigmoid(edge_vgg_3.fuse)
                print(edge_vgg_1, edge_vgg_3)
                edge_1 = tf.reshape(edge_1,
                                    [-1, input_pad.get_shape().as_list()[1], input_pad.get_shape().as_list()[2],
                                     1])
                edge_3 = tf.reshape(edge_3,
                                    [-1, input_pad.get_shape().as_list()[1], input_pad.get_shape().as_list()[2],
                                     1])
                print(edge_vgg_1, edge_vgg_3)
                with tf.variable_scope("Cycle_DVF"):
                    # Prepare model.
                    model = Voxel_flow_model(is_train=False)
                    prediction = model.inference(tf.concat([input_pad, edge_1, edge_3], 3))[0]
                sess = tf.Session()
                restorer = tf.train.Saver()
                restorer.restore(sess,
                                 os.path.join(ckptsPath, ckpt_file_name))

                feed_dict = {input_placeholder: np.concatenate((data_frame1, data_frame3), 3)}
                # Run single step update.
                prediction_np = sess.run(prediction, feed_dict=feed_dict)

                output = prediction_np[-1, pad_up:adatptive_H - pad_bot, pad_left:adatptive_W - pad_right, :]
                output = np.round(((output + 1.0) * 255.0 / 2.0)).astype(np.uint8)
                output = np.dstack((output[:, :, 2], output[:, :, 1], output[:, :, 0]))

        # cv2.imwrite(out, output)
        print("this is the output shape:",output.shape)

        output = output[-1, pad_up:adatptive_H - pad_bot, pad_left:adatptive_W - pad_right, :]
        output = np.round(((output + 1.0) * 255.0 / 2.0)).astype(np.uint8)
        output = np.dstack((output[:, :, 2], output[:, :, 1], output[:, :, 0]))
        print("this is the second output",output.shape)
        img = output[-1, int((output.shape[1] - height) / 2):height + int((output.shape[1] - height) / 2),
                int((output.shape[2] - width) / 2):width + int((output.shape[2] - width) / 2), -1]
        cv2.imwrite(os.path.join(out, ckpt_file_name + "_PeopleOnStreet_2560x1600_ttt.png"), img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
        out_file_path = os.path.join(out, ckpt_file_name + "_PeopleOnStreet_2560x1600_ttt.yuv")
        print("this is second img", img)
        cv2.VideoWriter(out_file_path,
                        cv2.VideoWriter_fourcc('I', '4', '2', '0'), 24, (width, height)).write(
            img)
        print("***************************************************one frame done*********************************************************")





if __name__ == '__main__':
    ckptsPath = r"CyclicGen_checkpoints_stage1_onlyCyc_20201201_3w_TCR"
    first = r'E:\TCR\CycNet\CyclicGen-master\result-yuv\Rec\BQMall_qp37_ra_832x480_0.yuv'
    second = r'E:\TCR\CycNet\CyclicGen-master\result-yuv\Rec\BQMall_qp37_ra_832x480_2.yuv'

    out = r'E:\TCR\CycNet\CyclicGen-master\test_LDP'

    org_frame_path = r"E:\TCR\CycNet\CyclicGen-master\result-yuv\Org\BQMall_Org_C_832x480_1.yuv"
    test(first, second, out)

    # printPsnr(sorted(psnrlist), org_frame_path)

    # for i in range(48):
    #     if i % 2 == 0:
    #         first = r'F:\gx\CyclicGen-master\yuv\BQMall_qp37_ra_832x480_' + str(i) + '.yuv'
    #         second = r'F:\gx\CyclicGen-master\yuv\BQMall_qp37_ra_832x480_' + str(i+2) + '.yuv'
    #         out = r'F:\gx\CyclicGen-master\yuv\1\BQMall_qp37_ra_832x480_' + str(i+1) + '.yuv'
    #         test(first, second, out)
