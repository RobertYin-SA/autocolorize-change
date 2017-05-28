# -*- coding: utf-8 -*-
"""
author : yinboya

the edited file of https://github.com/pavelgonchar/colornet
"""
import tensorflow as tf
import numpy as np
from batchnorm import ConvolutionalBatchNormalizer
from data_utils import *



class colornet(object):
    def __init__(self, init_refer_graph_path, train_filenames, batch_size, num_epochs):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.uv = tf.placeholder(tf.uint8, name='uv')

        # f=open("/home/yinboya/STUDY/DeepLearning/project/colornet-master/vgg/vgg16-20160129.tfmodel", mode='rb')
        graph_file = open(init_refer_graph_path, mode='rb')
        fileContent = graph_file.read()

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)

        with tf.variable_scope('colornet'):
            # Store layers weight
            weights = {
                # 1x1 conv, 512 inputs, 256 outputs
                'wc1': tf.Variable(tf.truncated_normal([1, 1, 512, 256], stddev=0.01)),
                # 3x3 conv, 512 inputs, 128 outputs
                'wc2': tf.Variable(tf.truncated_normal([3, 3, 256, 128], stddev=0.01)),
                # 3x3 conv, 256 inputs, 64 outputs
                'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 64], stddev=0.01)),
                # 3x3 conv, 128 inputs, 3 outputs
                'wc4': tf.Variable(tf.truncated_normal([3, 3, 64, 3], stddev=0.01)),
                # 3x3 conv, 6 inputs, 3 outputs
                'wc5': tf.Variable(tf.truncated_normal([3, 3, 3, 3], stddev=0.01)),
                # 3x3 conv, 3 inputs, 2 outputs
                'wc6': tf.Variable(tf.truncated_normal([3, 3, 3, 2], stddev=0.01)),
            }

        # train_filenames = sorted(glob.glob("/home/yinboya/STUDY/DeepLearning/project/colornet-master/test/*.jpg"))
        self.colorimage = input_pipeline(train_filenames, batch_size, num_epochs=num_epochs)
        self.colorimage_yuv = rgb2yuv(self.colorimage)

        self.grayscale = tf.image.rgb_to_grayscale(self.colorimage)
        self.grayscale_rgb = tf.image.grayscale_to_rgb(self.grayscale)
        self.grayscale_yuv = rgb2yuv(self.grayscale_rgb)
        self.grayscale = tf.concat([self.grayscale, self.grayscale, self.grayscale], 3)

        tf.import_graph_def(graph_def, input_map={"images": self.grayscale})

        graph = tf.get_default_graph()

        with tf.variable_scope('vgg'):
            conv1_2 = graph.get_tensor_by_name("import/conv1_2/Relu:0")
            conv2_2 = graph.get_tensor_by_name("import/conv2_2/Relu:0")
            conv3_3 = graph.get_tensor_by_name("import/conv3_3/Relu:0")
            conv4_3 = graph.get_tensor_by_name("import/conv4_3/Relu:0")


        tensors = {
            "conv1_2": conv1_2,
            "conv2_2": conv2_2,
            "conv3_3": conv3_3,
            "conv4_3": conv4_3,
            "grayscale": self.grayscale,
            "weights": weights
        }


        # Construct model
        self.pred = self.colornet(tensors)
        self.pred_yuv = tf.concat([tf.split(self.grayscale_yuv, 3, 3)[0], self.pred],3)
        self.pred_rgb = yuv2rgb(self.pred_yuv)

        self.loss = tf.square(tf.subtract(self.pred, tf.concat([tf.split(self.colorimage_yuv, 3, 3)[1], tf.split(self.colorimage_yuv, 3, 3)[2]],3)))

        if uv == 1:
            self.loss = tf.split(self.loss, 2, 3)[0]
        elif uv == 2:
            self.loss = tf.split(self.loss, 2, 3)[1]
        else:
            self.loss = (tf.split(self.loss, 2, 3)[0] + tf.split(self.loss, 2, 3)[1]) / 2

        if phase_train is not None:
            optimizer = tf.train.GradientDescentOptimizer(0.0001)
            self.opt = optimizer.minimize(self.loss, global_step=self.global_step, gate_gradients=optimizer.GATE_NONE)

        # Summaries
        tf.summary.histogram("weights1", weights["wc1"])
        tf.summary.histogram("weights2", weights["wc2"])
        tf.summary.histogram("weights3", weights["wc3"])
        tf.summary.histogram("weights4", weights["wc4"])
        tf.summary.histogram("weights5", weights["wc5"])
        tf.summary.histogram("weights6", weights["wc6"])
        tf.summary.histogram("instant_loss", tf.reduce_mean(self.loss))
        tf.summary.image("colorimage", self.colorimage, max_outputs=1)
        tf.summary.image("pred_rgb", self.pred_rgb, max_outputs=1)
        tf.summary.image("grayscale", self.grayscale_rgb, max_outputs=1)


    def batch_norm(self, x, depth, phase_train):
        with tf.variable_scope('batchnorm'):
            ewma = tf.train.ExponentialMovingAverage(decay=0.9999)
            bn = ConvolutionalBatchNormalizer(depth, 0.001, ewma, True)
            update_assignments = bn.get_assigner()
            x = bn.normalize(x, train=phase_train)
        return x


    def conv2d(self, _X, w, sigmoid=False, bn=False):
        with tf.variable_scope('conv2d'):
            _X = tf.nn.conv2d(_X, w, [1, 1, 1, 1], 'SAME')
            if bn:
                _X = self.batch_norm(_X, w.get_shape()[3], phase_train)
            if sigmoid:
                return tf.sigmoid(_X)
            else:
                _X = tf.nn.relu(_X)
                return tf.maximum(0.01 * _X, _X)


    def colornet(self, _tensors):
        """
        Network architecture http://tinyclouds.org/colorize/residual_encoder.png
        """
        with tf.variable_scope('colornet'):
            # Bx28x28x512 -> batch norm -> 1x1 conv = Bx28x28x256
            conv1 = tf.nn.relu(tf.nn.conv2d(self.batch_norm(_tensors["conv4_3"], 512, phase_train),
                _tensors["weights"]["wc1"], [1, 1, 1, 1], 'SAME'))
            # upscale to 56x56x256
            conv1 = tf.image.resize_bilinear(conv1, (56, 56))
            conv1 = tf.add(conv1, self.batch_norm(
                _tensors["conv3_3"], 256, phase_train))

            # Bx56x56x256-> 3x3 conv = Bx56x56x128
            conv2 = self.conv2d(conv1, _tensors["weights"]['wc2'], sigmoid=False, bn=True)
            # upscale to 112x112x128
            conv2 = tf.image.resize_bilinear(conv2, (112, 112))
            conv2 = tf.add(conv2, self.batch_norm(
                _tensors["conv2_2"], 128, phase_train))

            # Bx112x112x128 -> 3x3 conv = Bx112x112x64
            conv3 = self.conv2d(conv2, _tensors["weights"]['wc3'], sigmoid=False, bn=True)
            # upscale to Bx224x224x64
            conv3 = tf.image.resize_bilinear(conv3, (224, 224))
            conv3 = tf.add(conv3, self.batch_norm(_tensors["conv1_2"], 64, phase_train))

            # Bx224x224x64 -> 3x3 conv = Bx224x224x3
            conv4 = self.conv2d(conv3, _tensors["weights"]['wc4'], sigmoid=False, bn=True)
            conv4 = tf.add(conv4, self.batch_norm(
                _tensors["grayscale"], 3, phase_train))

            # Bx224x224x3 -> 3x3 conv = Bx224x224x3
            conv5 = self.conv2d(conv4, _tensors["weights"]['wc5'], sigmoid=False, bn=True)
            # Bx224x224x3 -> 3x3 conv = Bx224x224x2
            conv6 = self.conv2d(conv5, _tensors["weights"]['wc6'], sigmoid=True, bn=True)

        return conv6




