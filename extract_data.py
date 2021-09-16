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

import numpy as np
import cv2
import tensorflow as tf
import copy
import json

def extract_data(training_dataset_filepath, testing_dataset_filepath, num_test_images):
    train_data = np.genfromtxt(training_dataset_filepath, delimiter=',')
    train_data=np.delete(train_data, 0, 0)
    train_label=train_data[:,0]
    train_data=np.delete(train_data, 0, 1)

    testing_data = np.genfromtxt(testing_dataset_filepath, delimiter=',')
    testing_data=np.delete(testing_data, 0, 0)
    testing_label=testing_data[:,0]
    testing_data=np.delete(testing_data, 0, 1)

    if (num_test_images>0):
        #Generate Test Images
        print("Generating Test Images")
        result_dict=dict()
        #Note there is no J or Z
        result_guide=["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
        for i in range(len(testing_data[:num_test_images])):
            output_letter=result_guide[int(testing_label[i])]
            img_file='deploy/images/testimage_'+str(i)+'.png'
            result_dict.update({'testimage_'+str(i): output_letter})
            write_data=copy.copy(testing_data[i])
            write_data=write_data.reshape(28, 28, 1).astype('uint8')
            img = cv2.cvtColor(write_data, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(img_file, img)
        with open("deploy/resultguide.json", 'w') as outfile:
            json.dump(result_dict, outfile)


    train_data = train_data.reshape(27455, 28, 28, 1).astype('float32') / 255
    testing_data = testing_data.reshape(7172 , 28, 28, 1).astype('float32') / 255

    train_label = train_label.astype('float32')
    testing_label = testing_label.astype('float32')
            
    val_data = train_data[-4000:]
    val_label = train_label[-4000:]
    train_data = train_data[:-4000]
    train_label = train_label[:-4000]

    # one-hot encode the labels
    train_label = tf.keras.utils.to_categorical(train_label, num_classes=25)
    testing_label = tf.keras.utils.to_categorical(testing_label, num_classes=25)
    val_label = tf.keras.utils.to_categorical(val_label, num_classes=25)
    
    return [train_data, train_label, val_data, val_label, testing_data, testing_label]