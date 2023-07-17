"""
Generic setup of the data sources and the model training. 

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
and also on 
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

"""

#import keras
import CifarTernary
import sklearn.metrics as metrics
from keras.datasets       import mnist, cifar10
from keras.models         import Sequential
from keras.layers         import Dense, Dropout, Flatten, BatchNormalization, regularizers
from keras.utils.np_utils import to_categorical
from keras.callbacks      import EarlyStopping, Callback
from keras.layers         import Conv2D, MaxPooling2D, AveragePooling2D
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras                import backend as K
import tensorflow as tf
K.set_image_dim_ordering('tf')
import logging

# Helper: Early stopping.
early_stopper = EarlyStopping( monitor='val_acc', mode='max', patience=10, min_delta=0.001 )
#monitor='val_acc', min_delta=.01, patience=3, verbose=0, mode='auto'

#patience=5)
#monitor='val_loss',patience=2,verbose=0
#In your case, you can see that your training loss is not dropping - which means you are learning nothing after each epoch. 
#It look like there's nothing to learn in this model, aside from some trivial linear-like fit or cutoff value.

def get_cifar10_mlp():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes  = 10 #dataset dependent 
    batch_size  = 64
    epochs      = 4
    input_shape = (3072,) #because it's RGB

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test  = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

def get_cifar10_cnn():

    nb_classes = 10 #dataset dependent
    batch_size = 64
    epochs     = 14
  
    
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test,  nb_classes)

    #x._train shape: (50000, 32, 32, 3)
    #input shape (32, 32, 3)
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    #input_shape = (32, 32, 3)
    input_shape = x_test.shape[1:]

    #print('x_train shape:', x_train.shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')
    #print('input shape', input_shape)
   
    x_train = x_train.astype('float16')
    x_test  = x_test.astype('float16')
    x_train /= 255
    x_test  /= 255 
       
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

def get_mnist_mlp():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes  = 10 #dataset dependent 
    batch_size  = 64
    epochs      = 4
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test  = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

def get_mnist_cnn():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10 #dataset dependent
    batch_size = 128
    epochs     = 60
    
    # Input image dimensions
    img_rows, img_cols = 28, 28

    # Get the data.
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
   

    return (nb_classes, batch_size, epochs)

def compile_model_mlp(geneparam, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers  = geneparam['nb_layers' ]
    nb_neurons = geneparam['nb_neurons']
    activation = geneparam['activation']
    optimizer  = geneparam['optimizer' ]

    logging.info("Architecture:%d,%s,%s,%d" % (nb_neurons, activation, optimizer, nb_layers))

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout for each layer

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer=optimizer,
                    metrics=['accuracy'])

    return model

def compile_model_cnn(geneparam):
    """Compile a sequential model.

    Args:
        genome (dict): the parameters of the genome

    Returns:
        a compiled network.

    """
    # Get our network parameters.
   
    activationL = geneparam['activationL']
    activationR = geneparam['activationR']

    logging.info("Architecture: %s, %s" %(activationL, activationR))

    
    ##################################################################################33
    nb_classes = 10
    img_rows, img_cols = 32, 32
    img_channels = 3
    epochs1= 15
    batch_size1=128
    acc =CifarTernary.cifar_10(batch_size1=batch_size1, epochs=epochs1, activation_L=activationL, activation_R=activationR)	
     
    
###########################################################################################################################

    
    print("Model created: acccccccccccc=%.4f" %acc)
    return acc

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def train_and_score(geneparam, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    #K.set_image_dim_ordering('th')
    logging.info("Getting Keras datasets")

    if dataset   == 'cifar10_mlp':
        nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_cifar10_mlp()
    elif dataset == 'cifar10_cnn':
        nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_cifar10_cnn()
    elif dataset == 'mnist_mlp':
        nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_mnist_mlp()
    elif dataset == 'mnist_cnn':
        nb_classes, batch_size, epochs = get_mnist_cnn()

    logging.info("Compling Keras model")
    acc = 0.0
    if dataset   == 'cifar10_mlp':
        model = compile_model_mlp(geneparam, nb_classes, input_shape)
    elif dataset == 'cifar10_cnn':
        acc = compile_model_cnn(geneparam)
    elif dataset == 'mnist_mlp':
        model = compile_model_mlp(geneparam, nb_classes, input_shape)
    elif dataset == 'mnist_cnn':
        acc  = compile_model_cnn(geneparam)



   
    #we do not care about keeping any of this in memory - 
    #we just need to know the final scores and the architecture
    
    return acc # 1 is accuracy. 0 is loss.
