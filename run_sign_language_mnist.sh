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

#Create directory
mkdir train
mkdir keras2tf
mkdir freeze
mkdir quantize
mkdir deploy
mkdir deploy/images

#Creatge 

echo "Train keras model and convert to TF"
python3 main.py 

echo "Freeze Graph"
freeze_graph --input_graph=./train/tf_complete_model.pb \
    --input_checkpoint=./train/tfchkpt.ckpt \
    --input_binary=true \
    --output_graph=./freeze/frozen_graph.pb \
    --output_node_names=activation_4_1/Softmax

echo "Evaluate Frozen Graph"
python3 evaluate_accuracy.py \
   --graph=./freeze/frozen_graph.pb \
   --input_node=input_1_1 \
   --output_node=activation_4_1/Softmax \
   --batchsize=32


echo "Quantizing frozen graph"

vai_q_tensorflow --version

vai_q_tensorflow quantize \
        --input_frozen_graph=./freeze/frozen_graph.pb \
        --input_nodes=input_1_1 \
        --input_shapes=?,28,28,1 \
        --output_nodes=activation_4_1/Softmax  \
        --input_fn=image_input_fn.calib_input \
        --output_dir=quantize \
        --calib_iter=100

echo "Evaluate Quantized Graph"

python3 evaluate_accuracy.py \
   --graph=./quantize/quantize_eval_model.pb \
   --input_node=input_1_1 \
   --output_node=activation_4_1/Softmax \
   --batchsize=32

echo "Compiling"

# target board
BOARD=ZCU104
ARCH=/opt/vitis_ai/compiler/arch/dpuv2/${BOARD}/${BOARD}.json

vai_c_tensorflow \
       --frozen_pb=./quantize/deploy_model.pb \
       --arch=${ARCH} \
       --output_dir=launchmodel \
       --net_name=SignLanguageMNISTnet \
       --options    "{'mode':'normal'}" 

# copy elf to target folder
cp launchmodel/*.elf deploy/.
cp -r target/* deploy/.
echo "  Copied elf file(s) to target folder"


echo "Finished"
