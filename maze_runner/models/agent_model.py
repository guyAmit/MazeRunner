import numpy as np
import pickle
from tensorflow.keras.layers import (
    Conv2D, Flatten, Concatenate, Input, MaxPool2D, Dense, LSTM)
from tensorflow.keras import Model
from consts import MAX_STEPS


def build_cnn_model(img_size):
    img = Input(shape=(img_size, img_size, 1), name='img')
    lstm_features = Input(shape=(4,), name='lstm')
    iligal_move = Input(shape=(1,), name='iligal_move')

    inputs = {'img': img,
              'lstm': lstm_features,
              'iligal_move': iligal_move}

    y = Conv2D(filters=24, kernel_size=5, strides=1, padding='same')(img)
    y = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(y)
    y = Flatten()(y)
    x = Concatenate()([y, lstm_features, iligal_move])
    x = Dense(units=10, activation='tanh')(x)
    x = Dense(units=4, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)

    return model


def build_lstm_model(time_stamps, feature_number):
    lstm_input = Input(shape=(time_stamps, feature_number))
    x = LSTM(units=3)(lstm_input)
    x = Dense(units=4, activation='softmax')(x)
    model = Model(inputs=lstm_input, outputs=x)
    return model


def build_dense_model(feature_number):
    # directions0-3, end_near_indicator4-7,
    # times_visited8-11
    inputs = Input(shape=(feature_number,))
    x = Dense(units=20, activation='tanh')(inputs)
    x = Dense(units=10, activation='tanh')(x)
    x = Dense(units=4, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    return model


def convert_to_directions(pred):
    if pred == 0:
        return (1, 0)
    if pred == 1:
        return (-1, 0)
    if pred == 2:
        return (0, -1)
    if pred == 3:
        return (0, 1)
    return


class Agent_Model():

    def __init__(self, net_type, img_size):
        self.img_size = img_size
        self.net_type = net_type
        if net_type == 'cnn':
            self.model = build_cnn_model(img_size=img_size)
        else:
            self.model = build_dense_model(feature_number=12)

    def predict(self, lstm_featuers=None,
                end_near_indicator=None, img=None,
                iligal_move=None, times_visited=None):
        if self.net_type == 'cnn':
            img = img / 255
            inputs = {'img': img,
                      'lstm': lstm_featuers.reshape(1, 4),
                      'iligal_move': iligal_move}

            pred = self.model.predict(inputs)
        else:
            featuers = np.concatenate((lstm_featuers,
                                       end_near_indicator,
                                       times_visited)).reshape(1, -1)
            pred = self.model.predict(featuers)

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
                if layer.name[:4] == 'lstm':
                    weights.append(layer_weights[0])
                    weights.append(layer_weights[1])
                    weights.append(layer_weights[2].reshape(-1, 1))
                if layer.name[:5] == 'dense':
                    weights.append(layer_weights[0])
                    weights.append(layer_weights[1].reshape(-1, 1))
        return weights

    def set_weights(self, weights):
        counter = 0
        for layer in self.model.layers[1:]:
            layer_weights = layer.get_weights()
            if len(layer_weights) != 0:
                w = []
                if layer.name[:6] == 'conv2d':
                    fillters = weights[counter].reshape((5, 5, 1, 24))
                    w.append(fillters)
                    w.append(weights[counter+1].reshape(-1))
                    counter += 2
                if layer.name[:4] == 'lstm':
                    w.append(weights[counter])
                    w.append(weights[counter+1])
                    w.append(weights[counter+2].reshape(-1))
                    counter += 3
                if layer.name[:5] == 'dense':
                    w.append(weights[counter])
                    w.append(weights[counter+1].reshape(-1))
                    counter += 2
                layer.set_weights(w)

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.get_weights(), fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            weights = pickle.load(fp)
            self.set_weights(weights)
