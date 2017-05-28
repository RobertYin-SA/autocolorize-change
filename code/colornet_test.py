# -*- coding: utf-8 -*-
"""
author : Robert Yin

the edited file of https://github.com/pavelgonchar/colornet

usage:
    python colornet_test.py --test_filenames ../test --original_graph_path ../vgg/vgg16-20160129.tfmodel --batch_size 1 --num_epochs 1 --training_checkpoint ../model --output_path ../pre_pic
"""
import argparse

import tensorflow as tf
import numpy as np
import glob
import sys
from matplotlib import pyplot as plt
import colornet_model
from data_utils import *



def test(args):
    """
    Testing without training step
    """
    with tf.Graph().as_default():
        sess = tf.Session()
        # sess.run(init_op)

        test_filenames = sorted(glob.glob(args.test_filenames + "/*.jpg"))
        init_refer_graph_path = args.original_graph_path
        batch_size = int(args.batch_size)
        num_epochs = int(args.num_epochs)
        with sess.as_default():

            network = colornet_model.colornet(init_refer_graph_path,\
                    test_filenames, batch_size, num_epochs)

            # Saver.

            # Create the graph, etc.


            # Create a session for running operations in the Graph.
            # Initialize the variables.
            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            sess.run(init_op)
            saver = tf.train.Saver(tf.all_variables())
            # saver = tf.train.import_meta_graph(args.training_checkpoint + "/model.ckpt.meta")
            ckpt = tf.train.get_checkpoint_state(args.training_checkpoint)
            if ckpt and ckpt.model_checkpoint_path:
                print("Continue training from the model {}".format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)

            merged = tf.summary.merge_all()

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            count = 0
            try:
                while not coord.should_stop():
                    count += 1
                    pred_, pred_rgb_, colorimage_, grayscale_rgb_, cost, merged_ = sess.run(
                        [network.pred, network.pred_rgb, network.colorimage, network.grayscale_rgb, network.loss, merged],
                        feed_dict={network.phase_train: False, network.uv: 3})
                    summary_image = concat_images(grayscale_rgb_[0], pred_rgb_[0])
                    summary_image = concat_images(summary_image, colorimage_[0])
                    plt.imsave(args.output_path + "/" + str(count) + "_0.jpg", summary_image)

                    sys.stdout.flush()

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify arguments")
    parser.add_argument("--test_filenames", help="The train file path")
    parser.add_argument("--training_checkpoint", help="The checkpoint path during training")
    parser.add_argument("--original_graph_path", help="The original graph path of vgg16")
    parser.add_argument("--batch_size", help="Training batch size")
    parser.add_argument("--num_epochs", help="Training epochs. If in testing step, please use 1!")
    parser.add_argument("--output_path", help="output path of testing")
    args = parser.parse_args()
    test(args)

