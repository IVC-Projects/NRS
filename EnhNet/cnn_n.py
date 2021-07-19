import sys

if not hasattr(sys, 'argv'):
    sys.argv = ['']
import numpy as np
import tensorflow as tf
import os, time
from PIL import Image
from RWARN_V2 import model
#from VDSRx15_SE_VDSR import model
from UTILS import *
import time
import struct

I_QP37_modelpath = r"D:\HM_tensorflow\file\model\051601\0516-data1_139\0516-d1_139.ckpt"
model_set={'I_QP37':r'D:\HM_tensorflow\file\model\051601\0516-data1_139\0516-d1_139.ckpt',
            'I_QP37_0912':r"D:\konglingyi\trainWork\train_resiMV\checkpoints\test_train\test_train_099.ckpt",
           'I_QP37_sefcnn':r'D:\HM_tensorflow\file\model\qp37\tmp_I_set1257_model\VDSRx15_SE_qp37_newset1_570.ckpt',
           'I_QP37_091302':r"D:\konglingyi\trainWork\train_resiMV\checkpoints\train091302\train091302_169.ckpt",          #CNN*1
           'I_QP37_091501':r"D:\konglingyi\trainWork\train_resiMV\checkpoints\train091501\train091501_313473.91_229.ckpt",
           'I_QP37_091706':r"D:\konglingyi\trainWork\train_resiMV\checkpoints\train091706\train091706_332_156.65.ckpt",
           'I_QP37_091803':r"D:\konglingyi\trainWork\train_resiMV\checkpoints\train091801\train091801_108_154.73.ckpt",#CNN*2
            'I_QP37_091901':r"D:\konglingyi\trainWork\train_resiMV\checkpoints\train091901\train091901_160_155.32.ckpt",
'B_QP37_091804': r"D:\konglingyi\trainWork\train_resiMV\checkpoints\train091804\train091804_324_130.32.ckpt",
'SEFCNN_I_QP37': r"D:\HM_tensorflow\file\model\qp37\tmp_I_set1257_model\VDSRx15_SE_qp37_newset1_570.ckpt",
           }
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

input_tensor = None
shared_model = None
output_tensor = None
sess =None

def prepare_test_data(fileOrDir):
    original_ycbcr = []
    imgCbCr = []
    gt_y = []
    fileName_list = []
    # The input is a single file.
    if type(fileOrDir) is str:
        fileName_list.append(fileOrDir)

        # w, h = getWH(fileOrDir)
        # imgY = getYdata(fileOrDir, [w, h])
        imgY = c_getYdata(fileOrDir)
        imgY = normalize(imgY)

        imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
        original_ycbcr.append([imgY, imgCbCr])

    ##The input is one directory of test images.
    elif len(fileOrDir) == 1:
        fileName_list = load_file_list(fileOrDir)
        for path in fileName_list:
            # w, h = getWH(path)
            # imgY = getYdata(path, [w, h])
            imgY = c_getYdata(path)
            imgY = normalize(imgY)

            imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
            original_ycbcr.append([imgY, imgCbCr])

    ##The input is two directories, including ground truth.
    elif len(fileOrDir) == 2:

        fileName_list = load_file_list(fileOrDir[0])
        test_list = get_train_list(load_file_list(fileOrDir[0]), load_file_list(fileOrDir[1]))
        for pair in test_list:
            or_imgY = c_getYdata(pair[0])
            gt_imgY = c_getYdata(pair[1])

            # normalize
            or_imgY = normalize(or_imgY)

            or_imgY = np.resize(or_imgY, (1, or_imgY.shape[0], or_imgY.shape[1], 1))
            gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1], 1))

            ## act as a placeholder
            or_imgCbCr = 0
            original_ycbcr.append([or_imgY, or_imgCbCr])
            gt_y.append(gt_imgY)
    else:
        print("Invalid Inputs.")
        exit(0)

    return original_ycbcr, gt_y, fileName_list


def load_model(modelPath):
    print('----------------------------------load_model and init TF!----------------------------------',flush=True)
    global input_tensor,shared_model,output_tensor,sess
    tf.reset_default_graph()
    #ckpt = "0516-d1_139.ckpt"
    input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
    shared_model = tf.make_template('shared_model', model)
    output_tensor = shared_model(input_tensor)
    output_tensor = tf.clip_by_value(output_tensor, 0., 1.)
    output_tensor = tf.multiply(output_tensor, 255)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, modelPath)
    #print('Successfully load the pre-trained model!',flush=True)


def predict(fileOrDir):
    print('----------------------------------Predicting----------------------------------', flush=True)
    global input_tensor, shared_model, output_tensor,sess
    if (isinstance(fileOrDir, str)):
        original_ycbcr, gt_y, fileName_list = prepare_test_data(fileOrDir)
        imgY = original_ycbcr[0][0]
    else:
        fileOrDir = np.asarray(fileOrDir, dtype='float32')
        # fileOrDir = fileOrDir / 255
        imgY = normalize(np.reshape(fileOrDir, (1, len(fileOrDir), len(fileOrDir[0]), 1)))
    out = sess.run(output_tensor, feed_dict={input_tensor: imgY})
    out = np.reshape(out, (out.shape[1], out.shape[2]))
    out = np.around(out)
    out = out.astype('int')
    out = out.tolist()
    #saveImg(out)
    print('Successfully predict!', flush=True)
    return out


def modelI(inp):
    # tf.logging.warning("python, in I")
    # i = test_all_ckpt(I_MODEL_PATH, inp, 1)
    i = predict(inp)
    return i


def init(sliceType,QP):
    sliceType_list=['COMMON','I','B','P']
    mModel =sliceType_list[sliceType]+'_QP'+str(QP)
    #print('model name :',mModel,'  model path :',model_set[mModel])
    load_model(model_set['I_QP37_091901'])
    #load_model(model_set['I_QP37_sefcnn'])


def saveImg(file,inp):

    h, w = inp[0], inp[1]
    # tem = np.asarray(inp, dtype='uint8')
    # #np.save(r"H:\KONG\cnn_2K%f" % time.time(),tem)
    # tem = Image.fromarray(tem, 'L')
    # tem.save("D:/rec/FromPython%f.jpg" % time.time())
    with open(file,'wb') as fp:
        for line in inp:
            for i in line:
                #print(i)
                fp.write(struct.pack('B',i))

    print('image saved')


if __name__ == '__main__':
    inDir=r"H:\KONG\2K\q37"
    outDir=r"H:\KONG\2K\q37_SEFCNN_CNN1"
    #file = "D:/rec/out_FourPeople_1280x720_60.yuv"
    tf.reset_default_graph()
    load_model(model_set['SEFCNN_I_QP37'])
    for filename in [y for y in os.listdir(inDir) if os.path.isfile(os.path.join(inDir, y))]:
        print(filename)
        file=os.path.join(inDir, filename)
        rec = predict(file)
        #saveImg(os.path.join(outDir,filename.replace("out","cnn1")),rec)
        saveImg(os.path.join(outDir, 'cnn_1_'+filename), rec)
        #print((rec[0]))
        #tem=np.asarray(rec,dtype='uint8')
        #np.save(os.path.join(outDir,filename),tem)
        #saveImg(rec)

