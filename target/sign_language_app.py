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

'''Modifed by Beetlebox Limited for usage with the MNIST Sign Language Database

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

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart, xir
import os
import math
import threading
import time
import argparse
import json


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


'''
run CNN with batch
dpu: dpu runner
img: imagelist to be run
'''
def runDPU(dpu,img,batchSize,results,threadId,threadImages,startIdx):

    """get tensor"""
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    print(inputTensors)
    print(outputTensors)
    #tensorformat = dpu.get_tensor_format()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)
    #softmax = np.empty(outputSize)

    n_of_images = len(img)
    count = 0
    write_index = startIdx

    while count < n_of_images:

        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count
         
        """ prepare batch input/output """
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]
        
        """ init input image to input buffer """
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j,...] = img[(count+j) % n_of_images].reshape(input_ndim[1:])

        """ run with batch """
        job_id = dpu.execute_async(inputData,outputData)
        dpu.wait(job_id)
        np.set_printoptions(threshold=np.inf)

        """ calculate argmax over batch """
        for j in range(runSize):
            out_q[write_index] = np.argmax((outputData[0][j]))
            write_index += 1

        count = count + runSize

    return



def runApp(batchSize, threadnum, model, image_dir, custom_dir):

    """ create runner """
    #dpu = runner.Runner(meta_json)
    g = xir.Graph.deserialize(model)
    print(g)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threadnum):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    listImage=os.listdir(image_dir)

    customListImage=os.listdir(custom_dir)
    runTotal = len(listImage)+len(customListImage)

    global out_q
    out_q = [None] * runTotal

    """ pre-process all images """
    img = []
    for i in range(len(listImage)):
        image = cv2.imread(os.path.join(image_dir,listImage[i]), cv2.IMREAD_GRAYSCALE)
        image = image.reshape(28,28,1)
        image = image/255.0
        img.append(image)

    custom_img = []
    for i in range(len(customListImage)):
        image = cv2.imread(os.path.join(custom_dir,customListImage[i]), cv2.IMREAD_GRAYSCALE)
        image = image.reshape(28,28,1)
        image = image/255.0
        custom_img.append(image)


    """ make a list to hold results - each thread will write into it """
    results = [None] * len(img)
    custom_results = [None] * len(custom_img)


    """run with batch """
    threadAll = []
   
    threadImages=int(len(img)/threadnum)+1
    customThreadImages=int(len(custom_img)/threadnum)+1

    # set up the threads
    for i in range(threadnum):
        startIdx = i*threadImages
        if ( (len(listImage)-(i*threadImages)) > threadImages):
            endIdx=(i+1)*threadImages
        else:
            endIdx=len(listImage)
        t1 = threading.Thread(target=runDPU, args=(all_dpu_runners[i],img[startIdx:endIdx],batchSize,results,i,threadImages, startIdx))
        threadAll.append(t1)

    # set up the custom threads
    # for i in range(threadnum):
    #     startIdx = i*customThreadImages
    #     if ( (len(customListImage)-(i*customThreadImages)) > customThreadImages):
    #         endIdx=(i+1)*customThreadImages
    #     else:
    #         endIdx=len(customListImage)
    #     t2 = threading.Thread(target=runDPU, args=(dpu,custom_img[startIdx:endIdx],batchSize,custom_results,i,customThreadImages))
    #     threadAll.append(t2)

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print("Throughput: %.2f FPS" %fps)

    # post-processing - compare results to ground truth labels
    # ground truth labels are first part of image file name
    # Note no J or Z on purpose
    result_guide=["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
    correct=0
    wrong=0

    # print("Custom Image Predictions:")
    # for i in range(len(custom_img)):
    #     gt = customListImage[i].split('.')
    #     print("Custom Image: ", gt[0], " Predictions:", result_guide[custom_results[i]])


    with open('resultguide.json', 'r') as json_file:
        ground_truth= json.load(json_file)

    for i in range(len(listImage)):
        gt = listImage[i].split('.')
        print(out_q)
        ground_truth_value=ground_truth.get(gt[0])
        if (ground_truth_value==result_guide[results[i]]):
            correct+=1
            print(listImage[i], 'Correct { Ground Truth: ',ground_truth_value ,'Prediction: ', result_guide[results[i]], '}')
        else:
            wrong+=1
            print(listImage[i], 'Wrong { Ground Truth: ',ground_truth_value ,'Prediction: ', result_guide[results[i]], '}')

    #acc = (correct/len(listImage))*100
    #print('Correct:',correct,'Wrong:',wrong,'Accuracy: %.2f' %acc,'%')

    #del dpu

    return


def main():

    # command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model',
                    type=str,
                    required=True,
  	                help='Path of xmodel. No default, must be supplied by user.')
    ap.add_argument('-i', '--image_dir',
                    type=str,
                    default='images',
  	                help='Path of images folder. Default is ./images')
    ap.add_argument('-c', '--custom_dir',
                    type=str,
                    default='custom_images',
  	                help='Path of custom images folder. Default is ./custom_images')
    ap.add_argument('-t', '--threads',
                    type=int,
                    default=1,
  	                help='Number of threads. Default is 1')
    ap.add_argument('-b', '--batchsize',
                    type=int,
                    default=1,
  	                help='Input batchsize. Default is 1')
    args = ap.parse_args()


    runApp(args.batchsize, args.threads, args.model, args.image_dir, args.custom_dir)

    
if __name__ == '__main__':
    main()