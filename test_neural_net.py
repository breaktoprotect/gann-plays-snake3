# Supress warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import tensorflow as tf
#tf.get_logger().setLevel('INFO')

from keras.layers import Input, Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np

#* testing simple nn 
import simple_neural_network as snn


def keras_neural_net():
    model = Sequential()
    model.add(Dense(8, activation='relu',input_shape=(4,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))

    return model

def main_test():
    # Get 4, 8, 8, 4 fully connected model
    keras_model = keras_neural_net()

    keras_weights = keras_model.get_weights()

    for i, x in enumerate(keras_weights):
        for j, y in enumerate(keras_weights[i]):
            keras_weights[i][j] = (i+j)*0.02

    keras_model.set_weights(keras_weights)

    #print(keras_model.get_weights())

    X1 = np.array([1,0,0,0]).reshape(1,4)
    X2 = np.array([0,1,0,0]).reshape(1,4)
    X3 = np.array([0,0,1,0]).reshape(1,4)
    X4 = np.array([0,0,0,1]).reshape(1,4)
    print(keras_model.predict(X1))
    print(keras_model.predict(X2))
    print(keras_model.predict(X3))
    print(keras_model.predict(X4))

    print("")

    snn_model = snn.NeuralNet(4,8,8,4)

    snn_weights = snn_model.get_weights_biases()

    for i, x in enumerate(snn_weights):
        for j, y in enumerate(snn_weights[i]):
            snn_weights[i][j] = (i+j)*0.02

    #print(snn_weights)

    print(snn_model.feed_forward([1,0,0,0]))
    print(snn_model.feed_forward([0,1,0,0]))
    print(snn_model.feed_forward([0,0,1,0]))
    print(snn_model.feed_forward([0,0,0,1]))


if __name__ == "__main__":
    main_test()
