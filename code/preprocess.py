import numpy as np
from Beras.onehot import OneHotEncoder
from Beras.core import Tensor
from tensorflow.keras import datasets


def load_and_preprocess_data():
    '''This is where we load in and preprocess our data! We load in the data 
        for you but you'll need to flatten the images, normalize the values and 
        convert the input images from numpy arrays into tensors
    Return the preprocessed training and testing data and labels!'''

    # Load in the training and testing data from the MNIST dataset
    (train_inputs, train_labels), (test_inputs,
                                   test_labels) = datasets.mnist.load_data()
    # TODO: Flatten (reshape) and normalize the inputs
    # Hint: train and test inputs are numpy arrays so you can use np methods on them!
    train_inputs = train_inputs.reshape(-1,
                                        train_inputs.shape[1]*train_inputs.shape[2]) / 255
    test_inputs = test_inputs.reshape(-1,
                                      test_inputs.shape[1]*test_inputs.shape[2]) / 255

    # TODO: Convert all of the data into Tensors. The constructor is already
    # written for you in Beras/core.py and we import it in line 3
    train_inputs = Tensor(train_inputs)
    train_labels = Tensor(train_labels)
    test_inputs = Tensor(test_inputs)
    test_labels = Tensor(test_labels)

    return train_inputs, train_labels, test_inputs, test_labels
