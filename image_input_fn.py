'''
 Copyright 2020 Xilinx Inc.

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

import os
import cv2
import numpy as np

testing_data = './sign_mnist_test/sign_mnist_test.csv'
calib_batch_size = 32

testing_data = np.genfromtxt(testing_data, delimiter=',')
testing_data=np.delete(testing_data, 0, 0)
testing_label=testing_data[:,0]
testing_data=np.delete(testing_data, 0, 1)
#testing_data = testing_data.reshape(7172 , 28, 28, 1).astype('float32') / 255

np.savetxt('./quantize/quant_calib.csv', testing_data, delimiter=',')
calib_image_data='./quantize/quant_calib.csv'

def calib_input(iter):
    data = np.loadtxt('./quantize/quant_calib.csv', delimiter=',')
    current_iteration=iter * calib_batch_size
    batch_data=data[current_iteration:current_iteration+calib_batch_size]
    batch_data = batch_data.reshape(calib_batch_size , 28, 28, 1).astype('float32') / 255
    return {"input_1_1": batch_data}
calib_input(1)
