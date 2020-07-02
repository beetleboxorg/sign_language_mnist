# FPGA Vitis AI Tutorial using Tensorflow and Keras

![Image of Sign Language MNIST](https://github.com/beetleboxorg/sign_language_mnist/blob/master/sign_language_cover_square.jpg)
A beginners guide to getting started with AI on FPGAs for embedded systems. This tutorial uses Xilinx Zynq series FPGAs and the Xilinx Vitis AI tool as well as Tensorflow and Keras. The tutorials accompanying this code can be found on the Beetlebox website or on our github.io:

Multi-Part Series:
1.[Introduction](https://beetlebox.org/vitis-ai-using-tensorflow-and-keras-tutorial-part-1/)

2.[Getting Started](https://beetlebox.org/vitis-ai-using-tensorflow-and-keras-tutorial-part-2/)

3.[Transforming Kaggle Data and Convolutional Neural Networks (CNNs)](https://beetlebox.org/vitis-ai-using-tensorflow-and-keras-tutorial-part-3/)

Github.io (Coming Soon):

The tutorials are focused on Sign Language recognition using Vitis AI to translate models built in Tensorflow and Kaggle, explaining both the theory of why and how we use FPGAs for AI and the practise of implementing it. The dataset was chosen because it is small enough to allow for quick training on CPUs.

## Sign Language Recognition
This tutorial uses the [Sign Language MNIST dataset from Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist). It consists of the alphabet represented in American Sign Language (excluding J and Z which require motion to represent). It is designed as a drop-in replacement of the famous MNIST dataset and uses the same 28x28 pixel format with 27,455 cases for training and 7172 cases for testing.  

## Tested Environment
* OS: Ubuntu 18.04
* Vitis AI version: V1.1
* FPGA used: Zynq Ultrascale+ series ZCU104
* Tensorflow version: 1.15
* Keras version: 2.2.5

## Installation Guide
* To run this, we will need to install Docker and Vitis AI, [instructions can be found here](https://github.com/Xilinx/Vitis-AI).
* Clone the repo into a folder
* In the repo download and extract the [Sign Language MNIST dataset from Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist)


## Quick Usage Guide
### Creating the Tensorflow model
To run the Sign Language Recognition. We need to launch the Vitis AI Docker from a terminal in our repo
```bash
cd <Cloned-directory>/sign_language_mnist
sudo systemctl enable docker
<Vitis-AI-Installation-Directory>/Vitis-AI/docker_run.sh xilinx/vitis-ai-cpu:latest

```
To test your own custom images, place an image inside **test**. Resizing is automatically performed. There are two test images already in there.

We then need to install Keras in the docker:
```bash
sudo su
conda activate vitis-ai-tensorflow
pip install keras==2.2.5
conda install -y pillow
exit
conda activate vitis-ai-tensorflow

```
We can then run the script to build it all:
```bash
cd sign_language_mnist
./run_sign_language_mnist.sh 

```
For an in-depth explanation of what the script does, [please see the tutorials](https://beetlebox.org/vitis-ai-using-tensorflow-and-keras-tutorial-part-1/)

### Running on the FPGA
Once we have our model, we then need to set up our FPGA. To do this we first need to flash an image containing all the hardware we need onto the FPGA. Fortunatly, Xilinx provides a pre-made one and instructions on how to flash an image in the [Vitis User Guide found here](https://www.xilinx.com/html_docs/vitis_ai/1_1/gum1570690244788.html). Ensure you can successfully boot and connect to the FPGA using SSH as outlined in the user guide. This may involve configuring the boards IP through ifconfig

```bash
ifconfig eth0 192.168.1.10 netmask 255.255.255.0

```

We then need to install the relevant libraries for DPU. Download the package [vitis-ai_v1.1_dnndk.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.1_dnndk.tar.gz) onto the host and then transfer onto the board through SCP. From the host:

```bash
scp <download-directory>/vitis-ai_v1.1_dnndk.tar.gz root@192.168.1.10:~/

```
On the board:

```bash
tar -xzvf vitis-ai_v1.1_dnndk.tar.gz
cd vitis-ai_v1.1_dnndk
./install.sh

```
We then need to copy the files over that we generated in the __deploy__ folder

```bash
scp <Cloned-directory>/sign_language_mnist/deploy root@192.168.1.10:~/

```

Finally we can run the file:
```bash
cd sign_language_mnist/deploy
source ./compile_shared.sh
python3 sign_language_app.py -t 1 -b 1 -j /home/root/deploy/dpuv2_rundir/

```
We should see a result like so:

```bash
Throughput: 1045.72 FPS
Custom Image Predictions:
Custom Image:  test_b  Predictions: U
Custom Image:  test_c  Predictions: F
testimage_9.png Correct { Ground Truth:  H Prediction:  H }
testimage_6.png Correct { Ground Truth:  L Prediction:  L }
testimage_5.png Correct { Ground Truth:  W Prediction:  W }
testimage_1.png Correct { Ground Truth:  F Prediction:  F }
testimage_2.png Correct { Ground Truth:  L Prediction:  L }
testimage_7.png Correct { Ground Truth:  P Prediction:  P }
testimage_4.png Correct { Ground Truth:  D Prediction:  D }
testimage_3.png Correct { Ground Truth:  A Prediction:  A }
testimage_0.png Correct { Ground Truth:  G Prediction:  G }
testimage_8.png Correct { Ground Truth:  D Prediction:  D }
Correct: 10 Wrong: 0 Accuracy: 100.00

```

