# -*- coding: utf-8 -*-
"""
author : yinboya

the edited file of https://github.com/pavelgonchar/colornet

usage:
    python colornet_train.py --train_filenames ../test --original_graph_path ../vgg/vgg16-20160129.tfmodel --batch_size 1 --num_epochs 1000 --checkpoint_path ../model
"""
import argparse

import tensorflow as tf
import numpy as np
import glob
import sys
from matplotlib import pyplot as plt
import colornet_model
from data_utils import *



def train(args):
    train_filenames = sorted(glob.glob(args.train_filenames + "/*.jpg"))
    init_refer_graph_path = args.original_graph_path
    batch_size = int(args.batch_size)
    num_epochs = int(args.num_epochs)

    network = colornet_model.colornet(init_refer_graph_path, train_filenames,
            batch_size, num_epochs)


    # Create the graph, etc.

    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    # Create a session for running operations in the Graph.
    sess = tf.Session()
    # Initialize the variables.
    # sess.run(init_op)
    sess.run(init_op)

    # Saver.
    saver = tf.train.Saver()

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("tb_log", sess.graph_def)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    try:
        while not coord.should_stop():
            # Run training steps
            training_opt = sess.run(network.opt, feed_dict={network.phase_train: True, network.uv: 1})
            training_opt = sess.run(network.opt, feed_dict={network.phase_train: True, network.uv: 2})

            step = sess.run(network.global_step)

            if step % 1 == 0:
                pred_, pred_rgb_, colorimage_, grayscale_rgb_, cost, merged_ =\
                        sess.run([network.pred, network.pred_rgb, network.colorimage,\
                        network.grayscale_rgb, network.loss, merged],\
                        feed_dict={network.phase_train: False, network.uv: 3})

                print({"step": step, "cost": np.mean(cost)})

                if step % 100 == 0:
                    summary_image = concat_images(grayscale_rgb_[0], pred_rgb_[0])
                    summary_image = concat_images(summary_image, colorimage_[0])
                    plt.imsave("pic/" + str(step) + "_0.jpg", summary_image)

                sys.stdout.flush()
                writer.add_summary(merged_, step)
                writer.flush()

            # Save the model in every n step
            if step % 100 == 0:
                save_path = saver.save(sess, args.checkpoint_path)
                print("Model saved in file: %s" % save_path)
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
    parser.add_argument("--train_filenames", help="The train file path")
    parser.add_argument("--original_graph_path", help="The original graph path of vgg16")
    parser.add_argument("--batch_size", help="Training batch size")
    parser.add_argument("--num_epochs", help="Training epochs")
    parser.add_argument("--checkpoint_path", help="Training checkpoint")
    args = parser.parse_args()
    train(args)

