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

import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.models import load_model

def keras2tf(keras_hdf5,tfckpt,tf_graph):
        
    # set learning phase for no training
    backend.set_learning_phase(0)

    loaded_model = load_model(keras_hdf5)

    print ('Keras model information:')
    print (' Input names :',loaded_model.inputs)
    print (' Output names:',loaded_model.outputs)
    print('-------------------------------------')

    # get the TensorFlow graph path, flilename and file extension
    tfgraph_path = './train'
    tfgraph_filename = 'tf_complete_model.pb'

    loaded_model.save('saved_model/tfgraph_filename')

    print ('TensorFlow information:')
    print (' Graph saved as     :',os.path.join(tfgraph_path,'saved_model/tfgraph_filename/saved_model.pb'))
    print('-------------------------------------')

    return