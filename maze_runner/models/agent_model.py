import numpy as np
import pickle

from consts import MAX_STEPS


def build_dense_model(feature_number):
    # directions0-3, curr_direction4-7,end_near_indicator8-11,
    # # times_visited12-15
    dense1 = (np.sqrt(2/(10+feature_number)))*np.random.randn(10,
                                                              feature_number)
    bias1 = np.sqrt(2/10)*np.random.randn(10)
    dense2 = (np.sqrt(2/(10+10)))*np.random.randn(10, 10)
    bias2 = np.sqrt(2/10)*np.random.randn(10)
    dense3 = (np.sqrt(2/(10+4)))*np.random.randn(4, 10)
    bias3 = np.sqrt(2/10)*np.random.randn(4)
    model = [(dense1, bias1), (dense2, bias2), (dense3, bias3)]
    return model


def _softmax(x):
    exp = np.exp(x)
    sums = np.sum(exp, axis=0, keepdims=True) + 1e-10
    return exp/sums


def _sigmond(x):
    return (1/(1+np.exp(-x)))


def _predict_dense(model, x):
    a = x
    for w, b in model[:-1]:
        a = w.dot(a) + b.reshape(-1, 1)
        # a[a < 0] = 0  #  relu
        # a = _sigmond(a)
        a = np.tanh(a)
    w, b = model[-1]
    a = w.dot(a) + b.reshape(-1, 1)
    return _softmax(a)


class Agent_Model():

    def __init__(self, algorithm_type, img_size):
        self.img_size = img_size
        self.algorithm_type = algorithm_type
        self.model = build_dense_model(feature_number=16)

    def predict(self, lstm_featuers=None, oposite_direction=None,
                end_near_indicator=None, img=None,
                iligal_move=None, times_visited=None):
        featuers = np.concatenate((lstm_featuers,
                                   oposite_direction,
                                   end_near_indicator,
                                   times_visited)).reshape(-1, 1)
        pred = _predict_dense(self.model, featuers)
        idx = np.argmax(pred, axis=0)
        return idx

    def get_weights(self):
        weights = []
        for w, b in self.model:
            weights.append(w.reshape(-1))
            weights.append(b.reshape(-1))
        return np.concatenate(weights)

    def set_weights(self, weights):
        counter = 0
        weights = np.array(weights)
        for i, (w, b) in enumerate(self.model):
            reshaped_w = weights[counter: counter +
                                 np.prod(w.shape)].reshape(w.shape)
            counter += np.prod(w.shape)
            reshaped_b = weights[counter: counter+np.prod(b.shape)]
            counter += np.prod(b.shape)
            self.model[i] = (reshaped_w, reshaped_b)

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.get_weights(), fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            weights = pickle.load(fp)
            self.set_weights(weights)
