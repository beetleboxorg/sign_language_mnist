'''
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''Modified by Beetlebox Limited for usage with the MNIST Sign Language Database

Modifications published under Apache License 2.0'''

'''
 Copyright 2020 Beetlebox Limited

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

import sys
import os
import argparse
import tensorflow as tf
import numpy as np
from keras import datasets, utils, layers, models, optimizers
from extract_data import extract_data
import tensorflow.contrib.decent_q


from progressbar import ProgressBar
from tensorflow.python.platform import gfile

#import tensorflow.contrib.decent_q


# reduce TensorFlow messages in console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def graph_eval(dataset_loc, input_graph_def, graph, input_node, output_node, batchsize):

    input_graph_def.ParseFromString(tf.gfile.GFile(graph, "rb").read())

    # Extact data from csv files again
    training_dataset_filepath='%ssign_mnist_train/sign_mnist_train.csv' % dataset_loc
    testing_dataset_filepath='%ssign_mnist_test/sign_mnist_test.csv' % dataset_loc
    train_data, train_label, val_data, val_label, testing_data, testing_label=extract_data(training_dataset_filepath, testing_dataset_filepath,0)

    total_batches = int(len(testing_data)/batchsize)
  
    #Import Graph
    tf.import_graph_def(input_graph_def,name = '')

    # Get input placeholders & tensors
    images_in = tf.get_default_graph().get_tensor_by_name(input_node+':0')
    labels = tf.placeholder(tf.int32,shape = [None,25])

    # get output tensors
    logits = tf.get_default_graph().get_tensor_by_name(output_node+':0')
    predicted_logit = tf.argmax(input=logits, axis=1, output_type=tf.int32)
    ground_truth_label = tf.argmax(labels, 1, output_type=tf.int32)

    # Define the metric and update operations
    tf_metric, tf_metric_update = tf.metrics.accuracy(labels=ground_truth_label,
                                                      predictions=predicted_logit,
                                                      name='acc')

    with tf.Session() as sess:
        progress = ProgressBar()
        
        sess.run(tf.initializers.global_variables())
        sess.run(tf.initializers.local_variables())

        # process all batches
        for i in progress(range(0,total_batches)):

            # fetch a batch from validation dataset
            x_batch, y_batch = testing_data[i*batchsize:i*batchsize+batchsize], \
                               testing_label[i*batchsize:i*batchsize+batchsize]

            # Run graph for accuracy node
            feed_dict={images_in: x_batch, labels: y_batch}
            acc = sess.run(tf_metric_update, feed_dict)

        print ('Graph accuracy with validation dataset: {:1.4f}'.format(acc))

    return



def main():

    argpar = argparse.ArgumentParser()
    argpar.add_argument('--dataset',
                    type=str,
                    default='./',
                    help='The directory where the dataset is held')
    argpar.add_argument('--graph',
                    type=str,
                    default='./freeze/frozen_graph.pb',
                    help='graph file (.pb) to be evaluated.')
    argpar.add_argument('--input_node',
                    type=str,
                    default='input_1_1',
                    help='input node.')
    argpar.add_argument('--output_node',
                    type=str,
                    default='activation_4_1/Softmax',
                    help='output node.')
    argpar.add_argument('-b', '--batchsize',
                    type=int,
                    default=32,
                    help='Evaluation batchsize, must be integer value. Default is 32') 
    args = argpar.parse_args()


    input_graph_def = tf.Graph().as_graph_def()
    graph_eval(args.dataset, input_graph_def, args.graph, args.input_node, args.output_node, args.batchsize)


if __name__ ==  "__main__":
    main()