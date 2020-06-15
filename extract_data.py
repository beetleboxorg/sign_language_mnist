import numpy as np
from keras import utils

def extract_data(training_dataset_filepath, testing_dataset_filepath):
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
    
    return [train_data, train_label, val_data, val_label, testing_data, testing_label]