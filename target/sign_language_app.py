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

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import json


def preprocess_fn(image_path):
    '''
    Image pre-processing.
    Opens image as grayscale then normalizes to range 0:1
    input arg: path of image file
    return: numpy array
    '''
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.reshape(28,28,1)
    image = image/255.0
    return image


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


def runDPU(id,start,dpu,img):

    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare batch input/output '''
        outputData = []
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])

        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData)
        dpu.wait(job_id)

        '''store output vectors '''
        for j in range(runSize):
            out_q[write_index] = np.argmax(outputData[0][j])
            write_index += 1
        count = count + runSize

def app(image_dir,threads,model,results_guide):

    listimage=os.listdir(image_dir)
    runTotal = len(listimage)

    with open(results_guide, 'r') as json_file:
        ground_truth_dict= json.load(json_file)

    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    ''' preprocess images '''
    print('Pre-processing',runTotal,'images...')
    img = []
    for i in range(runTotal):
        path = os.path.join(image_dir,listimage[i])
        img.append(preprocess_fn(path))

    '''run threads '''
    print('Starting',threads,'threads...')
    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start+(len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start=end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))


    ''' post-processing '''
    classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"] 
    correct = 0
    wrong = 0
    for i in range(len(out_q)):
        prediction = classes[out_q[i]]
        ground_truth, _ = listimage[i].split('_',1)
        gt = listimage[i].split('.')
        ground_truth_value=ground_truth_dict.get(gt[0])
        if (ground_truth_value==prediction):
            correct += 1
            print(listimage[i], 'Correct { Ground Truth: ',ground_truth_value ,'Prediction: ', prediction, '}')
        else:
            wrong += 1
            print(listimage[i], 'Wrong { Ground Truth: ',ground_truth_value ,'Prediction: ', prediction, '}')
    accuracy = correct/len(out_q)
    print('Correct:%d, Wrong:%d, Accuracy:%.4f' %(correct,wrong,accuracy))

    return



# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='images', help='Path to folder of images. Default is images')  
  ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='model_dir/customcnn.xmodel', help='Path of xmodel. Default is model_dir/customcnn.xmodel')
  ap.add_argument('-r', '--results',     type=str, default='./resultguide.json', help='Path to the result guide. Default is ./resultguide.json')
  ap.add_argument('-s', '--custom_results',     type=str, default='./test_resultguide.json', help='Path to the custom result guide. Default is ./test_resultguide.json')
  ap.add_argument('-c', '--custom_dir',     type=str, default='custom_images', help='Path of custom images folder. Default is ./custom_images')

  args = ap.parse_args()  
  
  print ('Command line options:')
  print (' --image_dir : ', args.image_dir)
  print (' --threads   : ', args.threads)
  print (' --model     : ', args.model)
  print (' --results     : ', args.results)
  print (' --custom_dir     : ', args.custom_dir)
  print (' --custom_results     : ', args.custom_results)


  # Standard Images
  app(args.image_dir,args.threads,args.model, args.results)
  # Custom Images
  app(args.custom_dir,args.threads,args.model, args.custom_results)

if __name__ == '__main__':
  main()

