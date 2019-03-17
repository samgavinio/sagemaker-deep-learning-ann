# In Legacy Mode there is no way to configure the epochs and batch size other than through the training_steps and evaluation_steps parameters
# See: https://github.com/aws/sagemaker-python-sdk/issues/561

import tensorflow as tf
import numpy as np
import os
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# Community documentation for Keras on SageMaker is few and far between. 
# * https://github.com/aws/sagemaker-python-sdk/tree/v1.12.0/src/sagemaker/tensorflow#tensorflow-sagemaker-estimators-and-models
# * https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_keras_cifar10/tensorflow_keras_CIFAR10.ipynb

INPUT_TENSOR_NAME = "inputs_input" # needs to match the name of the first layer + "_input"

def keras_model_fn(hyperparameters):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11, name = "inputs"))
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = hyperparameters["optimizer"], loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier

# Return the training set
def train_input_fn(training_dir, hyperparameters):    
    X_train = np.load(os.path.join(training_dir, 'train_X.npy'))
    y_train = np.load(os.path.join(training_dir, 'train_Y.npy'))

    return {INPUT_TENSOR_NAME: X_train}, y_train

# Return the testing set
def eval_input_fn(training_dir, hyperparameters):
    X_test = np.load(os.path.join(training_dir, 'test_X.npy'))
    y_test = np.load(os.path.join(training_dir, 'test_Y.npy'))

    return {INPUT_TENSOR_NAME: X_test}, y_test


# See: https://sagemaker-workshop.com/custom/algo.html#serving-the-model
# During an inference, the placeholder will be replaced. 
# The dictionary key INPUT_TENSOR_NAME must match what will be passed during the training and inference.
# The shape (1024, 11) means a 2D array of up to 1024 observations, each with 11 features
def serving_input_fn(hyperparameters):
    tensor = tf.placeholder(tf.float32, shape=(1024, 11))
    inputs = {INPUT_TENSOR_NAME: tensor}

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
