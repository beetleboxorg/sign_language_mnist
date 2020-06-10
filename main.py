import csv, argparse
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
from keras import datasets, utils, layers, models, optimizers
from nn_model import neural_network

def train_nn(dataset_loc):
    #Set Parameters
    DATASET_SIZE=27455
    BATCHSIZE=32
    EPOCHS=3
    LEARN_RATE=0.0001
    DECAY_RATE=1e-6    
    #Pre-processes data and trains the neural network
    
    #Open Training and test se
    training_debug_log = open("training_debug_log.txt", "w")
    testing_debug_log = open("training_debug_log.txt", "w")

    #Get the column names form the first row of the csv file
    training_dataset_filepath='%ssign_mnist_train/sign_mnist_train.csv' % dataset_loc
    testing_dataset_filepath='%ssign_mnist_test/sign_mnist_test.csv' % dataset_loc

    with open(training_dataset_filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        column_names = next(reader)

    train_data = np.genfromtxt(training_dataset_filepath, delimiter=',')
    train_data=np.delete(train_data, 0, 0)
    train_label=train_data[:,0]
    train_data=np.delete(train_data, 0, 1)

    testing_data = np.genfromtxt(testing_dataset_filepath, delimiter=',')
    testing_data=np.delete(testing_data, 0, 0)
    testing_label=testing_data[:,0]
    testing_data=np.delete(testing_data, 0, 1)

    train_data = train_data.reshape(27455, 28, 28, 1).astype('float32') / 255
    testing_data = testing_data.reshape(7172 , 28, 28, 1).astype('float32') / 255

    train_label = train_label.astype('float32')
    testing_label = testing_label.astype('float32')
    
    val_data = train_data[-4000:]
    val_label = train_label[-4000:]
    train_data = train_data[:-4000]
    train_label = train_label[:-4000]

    # one-hot encode the labels
    train_label = utils.to_categorical(train_label)
    testing_label = utils.to_categorical(testing_label)
    val_label = utils.to_categorical(val_label)

    model=neural_network()

    model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.Adam(lr=LEARN_RATE, decay=DECAY_RATE),
              metrics=['accuracy']
              )

    model.fit(train_data,
          train_label,
          batch_size=BATCHSIZE,
          shuffle=True,
          epochs=EPOCHS,
          validation_data=(val_data, val_label)
          )

    #Evaluate Model Accracy
    scores = model.evaluate(testing_data, 
                        testing_label,
                        batch_size=BATCHSIZE
                        )
    
    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])

    # save weights, model architecture & optimizer to an HDF5 format file
    model.save(os.path.join('./','keras_trained_model.h5'))
    print ('FINISHED!')

    print ('Convert Keras to TF')





def main():
    #The main file for everything that is to be run on host

    #Setup the 
    argpar = argparse.ArgumentParser()

    argpar.add_argument('--dataset',
                    type=str,
                    default='./',
                    help='The directory where the dataset is held')

    args = argpar.parse_args()  

    train_nn(args.dataset)

    

if __name__ ==  "__main__":
    main()