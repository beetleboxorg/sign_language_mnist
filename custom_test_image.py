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

'''Allows users to input custom images for final testing
    Input: Custom images from the test folder
    Output: A resized greyscale 28x28 version of the image'''

from os import listdir
from os.path import isfile, join
import numpy
import cv2

def custom_test_image():
    testpath='./test'
    onlyfiles = [ f for f in listdir(testpath) if isfile(join(testpath,f)) ]
    for n in range(0, len(onlyfiles)):
        image = cv2.imread( join(testpath,onlyfiles[n]) )
        dim = (28, 28)
        # resize image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        #Once greyscale it needs to be converted back into BGR format
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img_file='deploy/custom_images/' + onlyfiles[n]
        cv2.imwrite(img_file, img)

if __name__ ==  "__main__":
    custom_test_image()