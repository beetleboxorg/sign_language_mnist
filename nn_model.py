from keras import datasets, utils, layers, models, optimizers
from keras.callbacks import ModelCheckpoint

def neural_network():
    inputs = layers.Input(shape=(28, 28, 1))
    net = layers.Conv2D(28, kernel_size=(3, 3), padding='same')(inputs)
    net = layers.Activation('relu')(net)
    net = layers.BatchNormalization()(net)
    net = layers.MaxPooling2D(pool_size=(2,2))(net)

    net = layers.Conv2D(64, kernel_size=(3, 3), padding='same')(net)
    net = layers.Activation('relu')(net)
    net = layers.BatchNormalization()(net)
    net = layers.MaxPooling2D(pool_size=(2,2))(net)
    net = layers.Dropout(0.4)(net)

    net = layers.Flatten(input_shape=(28, 28,1))(net)
    net = layers.Dense(512)(net)
    net = layers.Activation('relu')(net)

    net = layers.Dropout(0.4)(net)
    net = layers.Dense(25)(net)
    prediction = layers.Activation('softmax')(net)

    model = models.Model(inputs=inputs, outputs=prediction)
    print(model.summary())

    return(model)

