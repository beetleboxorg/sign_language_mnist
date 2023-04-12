# Getting started with BeetleboxCI and Vitis AI: Sign Language MNIST

![Image of Sign Language MNIST](https://github.com/beetleboxorg/sign_language_mnist/blob/master/sign_language_cover_square.jpg)

This is a tutorial designed to show how to get started using BeetleboxCI and Vitis AI, in which we will cover the following: 

- **Designing a Neural Network** - Designing a simple neural network in Tensorflow and Keras.
- **Training a Neural Network** - We will show how this neural network can be trained through BeetleboxCI.
- **Converting the model and preparing for FPGAs** - Through BeetleboxCI we will also be converting the model for use on FPGAs as well as creating the files to run on the board.

[The full tutorial series may be found here on BeetleboxCI documentation](https://docs.beetleboxci.com/docs/tutorials/getting-stated-with-beetleboxci-and-vitis-ai-part-1)

[The accompanying git repository for this tutorial may be found here.](https://github.com/beetleboxorg/sign_language_mnist)

## Overview

A beginners guide to getting started with AI on FPGAs for embedded systems. This tutorial uses Xilinx Zynq series FPGAs and the Xilinx Vitis AI tool as well as Tensorflow and Keras. The tutorials accompanying this code can be found on the Beetlebox website or on our github.io:

The tutorials are focused on Sign Language recognition using Vitis AI to translate models built in Tensorflow and Kaggle, explaining both the theory of why and how we use FPGAs for AI and the practise of implementing it. The dataset was chosen because it is small enough to allow for quick training on CPUs.

## Tools we are using
 - **Vitis AI:** Vitis AI is Xilinx's development environment for AI on FPGAs.
 - **BeetleboxCI:** BeetleboxCI is the Continuous Integration software specifically designed for FPGA design. Our free version provides enough hours to run through this tutorial and [you can signup free from the website.](https://beetlebox.org)
 - **Github**: Github is a git repository service that can be connected to BeetleboxCI to create an automated workflow.

## Pre-requisites
 - A GitHub account.
 - [A kaggle account to download the required dataset](https://www.kaggle.com/datamunge/sign-language-mnist).

## Sign Language MNIST Dataset
This tutorial uses the [Sign Language MNIST dataset from Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist). It consists of the alphabet represented in American Sign Language (excluding J and Z which require motion to represent). It is designed as a drop-in replacement of the famous MNIST dataset and uses the same 28x28 pixel format with 27,455 cases for training and 7172 cases for testing.


## Tested Environment
 - OS: Ubuntu 18.04
 - Vitis AI version: V1.4
 - FPGA used: Zynq Ultrascale+ series ZCU104
 - Tensorflow version: 2.3
 - Keras version: 2.2.5

## Quick Start Guide
Let's get our AI up and running on BeetleboxCI  within five minutes.
1. Create a new git repository, which we will call <code>sign-language-mnist</code>.
2. [Clone the accompanying git repository for this.](https://github.com/beetleboxorg/sign_language_mnist)
```sh
git clone https://github.com/beetleboxorg/sign_language_mnist.git
```
3. Mirror push the cloned repository to your new repository.
```sh
cd sign_language_mnist
git push --mirror git@github.com:<yourgitaccount>/sign-language-mnist.git
```
4. Remove the cloned repository.
```sh
cd ..
rm -rf sign_language_mnist
```
5. Login to BeetleboxCI. In pipelines click on the button labelled <code>Create your first pipeline</code>.
6. In the following screen, fill in the following:
  - Project Name: sign_language_mnist
  - Repository URL: https://github.com/<yourgitaccount>/sign-language-mnist
  

7. Also fill in either the "Authentication settings" section OR the "SSH Authentication" section. You will need need to use username and token authentication if you chose the https URL or SSH authentication if you chose the SSH URL.   
  - Authentication Settings — Username: `<Github username>`
  - Authentication Settings — Password: `<Github personal access token>`
  
  
  - SSH Authentication — SSH private key  
  
8. Click proceed. You will now be redirected to the pipelines page where you can see the project that you just created. 

9. We now need to upload the dataset for training and testing. [Visit the kaggle page and download the dataset.](https://www.kaggle.com/datamunge/sign-language-mnist)
10. Go to the <code>Artifact Store</code> and click the button labelled <code>Upload your first artifact</code>.
11. In the file upload page, choose the file downloaded from Kaggle, which should be called <code>archive.zip</code>. Do not unzip it. Give the file the artifact type of <code>Miscellaneous</code>. Wait for the file to finish uploading, where you should be taken back to the Artifact Store:
![artifact-store](/img/tutorial/getting-started-with-beetleboxci-and-vitis-ai/sign-language-mnist-artifact-store.png)
12. In our projects, we should now see the sign language MNIST project. Click the play button to run the project.
![sign-language-mnist-run-project](/img/tutorial/getting-started-with-beetleboxci-and-vitis-ai/sign-language-mnist-run-project.png)
13. After a few minutes, the project should succesfully complete.
![sign-language-mnist-complete-project](/img/tutorial/getting-started-with-beetleboxci-and-vitis-ai/sign-language-mnist-complete-project.png)
14. The files needed to run this project on our FPGA are then stored in the artifact store.
![sign-language-mnist-complete-artifact-store](/img/tutorial/getting-started-with-beetleboxci-and-vitis-ai/sign-language-mnist-complete-artifact-store.png)

We have setup our code and data, trained our neural network, converted the model and prepared for use on a FPGA, all on a single pipeline.


## Running on the FPGA
Once we have downloaded our model from BeetleboxCI, we then need to set up our FPGA. To do this we first need to flash an image containing all the hardware we need onto the FPGA. Fortunatly, Xilinx provides a pre-made one and instructions on how to flash an image in the [Vitis User Guide found here](https://www.xilinx.com/html_docs/vitis_ai/1_1/gum1570690244788.html).
We need to ensure we can successfully boot and connect to the FPGA using SSH as outlined in the user guide. This may involve configuring the boards IP through ifconfig:

```bash
ifconfig eth0 192.168.1.10 netmask 255.255.255.0

```

We then need to copy the files over that we generated in the __deploy__ folder

```bash
scp <Cloned-directory>/sign_language_mnist/deploy root@192.168.1.10:~/

```

Finally we can run the file:
```bash
cd deploy
python3 sign_language_app.py --model sign_language_mnist.xmodel --image_dir images --threads 1 -s ./test_resultguide.json

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
