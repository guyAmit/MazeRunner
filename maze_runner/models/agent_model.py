import numpy as np
import pickle
from tensorflow.keras.layers import (
    Conv2D, Flatten, Concatenate, Input, MaxPool2D, Dense)
from tensorflow.keras import Model


def build_model(img_size):
    img = Input(shape=(img_size, img_size, 1))
    iligal_move = Input(shape=(1,))
    dead_end = Input(shape=(1,))
    y = Conv2D(filters=10, kernel_size=5, strides=1, padding='same')(img)
    y = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(y)
    y = Flatten()(y)
    x = Concatenate()([y, iligal_move, dead_end])
    x = Dense(units=10, activation='relu')(x)
    x = Dense(units=4, activation='softmax')(x)
    model = Model(inputs=[img, iligal_move, dead_end], outputs=x)

    return model


def convert_to_directions(pred):
    if pred == 0:
        return (1, 0)
    if pred == 1:
        return (-1, 0)
    if pred == 2:
        return (0, 1)
    if pred == 3:
        return (0, -1)
    return


class Agent_Model():

    def __init__(self, img_size):
        self.img_size = img_size
        self.model = build_model(img_size=img_size)

    def predict(self, img, iligal_move, dead_end):
        img = img / 255
        pred = self.model.predict([img, iligal_move, dead_end])
        idx = np.argmax(pred, axis=1)
        return convert_to_directions(idx)

    def get_weights(self):
        weights = []
        for layer in self.model.layers[1:]:
            layer_weights = layer.get_weights()
            if len(layer_weights) != 0:
                if layer.name[:6] == 'conv2d':
                    s = layer_weights[0].shape
                    weights.append(layer_weights[
                        0].reshape(s[3]*s[0], s[0]))
                    weights.append(layer_weights[1].reshape(-1, 1))
                if layer.name[:5] == 'dense':
                    weights.append(layer_weights[0])
                    weights.append(layer_weights[1].reshape(-1, 1))
        return weights

    def set_weights(self, weights):
        counter = 0
        for layer in self.model.layers[1:]:
            layer_weights = layer.get_weights()
            if len(layer_weights) != 0:
                if layer.name == 'conv2d':
                    fillters = weights[counter].reshape((5, 5, 1, 10))
                    weights[counter] = fillters
                layer.set_weights(
                    [weights[counter], weights[counter+1].reshape(-1)])
                counter += 2

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.get_weights(), fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            weights = pickle.load(fp)
            self.set_weights(weights)