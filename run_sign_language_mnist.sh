#!/bin/bash

# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Beetlebox Limited for usage with the MNIST Sign Language Database

# Modifications published under Apache License 2.0'''


#  Copyright 2020 Beetlebox Limited

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.




# remove previous results
rm -r train
rm -r keras2tf
rm -r freeze
rm -r quantize
rm -r deploy
rm -r dump


#Create directory
mkdir train
mkdir keras2tf
mkdir freeze
mkdir quantize
mkdir deploy
mkdir dump
mkdir deploy/images
mkdir deploy/custom_images

#Creatge 

echo "######## Resizing custom images for usage ########"
python3 custom_test_image.py

echo "######## Training keras model and converting to TF ########"
python3 train_nn.py 

echo "######## Compiling for the DPU ########"

vai_c_tensorflow2 \
    -m ./train/quantized_model.h5 \
    -a /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/arch/DPUCZDX8G/ZCU104/arch.json \
    -o ./deploy \
    -n sign_language_mnist

echo "######## Preparing deploy folder ########"
cp -r target/* deploy/.

echo "######## Finished ########"
