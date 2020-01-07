import numpy as np
import scipy.signal
import pickle


def conv2d(img, fillters, mode='same'):
    output = np.zeros((fillters.shape[0], *img.shape))
    for idx in range(fillters.shape[0]):
        output[idx] = scipy.signal.convolve2d(img, fillters[idx],
                                              mode=mode)
    return output


def build_model(fillters_number, dense_size, img_size, max_stride, max_size):
    fillters = (np.sqrt(2 / fillters_number) *
                np.random.randn(fillters_number, 3, 3))

    dense1 = np.sqrt((2 / (+dense_size))) * \
        np.random.randn(dense_size,
                        fillters_number*int(
                                           (int(
                                               (img_size-max_size+1) /
                                               max_stride)+1)**2))
    dense2 = np.sqrt((2/(dense_size+4)))*np.random.randn(4,
                                                         dense_size)
    weights = [fillters, dense1, dense2]
    return weights


def softmax(x):
    exps = np.exp(x)
    sums = np.sum(exps, axis=0, keepdims=True)
    return exps/sums


def pooling(feature_map, size=2, stride=2):
    pool_out = np.zeros((feature_map.shape[0],
                         int((feature_map.shape[1]-size+1)/stride)+1,
                         int((feature_map.shape[2]-size+1)/stride)+1))

    for map_num in range(feature_map.shape[0]):

        for r_idx, r in enumerate(np.arange(0,
                                            feature_map.shape[1]-size+1,
                                            stride)):
            for c_idx, c in enumerate(np.arange(0,
                                                feature_map.shape[2]-size+1,
                                                stride)):

                pool_out[map_num, r_idx, c_idx] = np.max(
                    feature_map[map_num,
                                r:r+size,
                                c:c+size])
    return pool_out


class Model():

    def __init__(self, fillters_number, dense_size, img_size):
        self.fillters_number = fillters_number
        self.dense_size = dense_size
        self.img_size = img_size
        self.weights = build_model(fillters_number,
                                   dense_size, img_size,
                                   max_stride=2, max_size=2)

    def predict(self, img):
        out1 = conv2d(img, self.weights[0], mode='same')
        out2 = pooling(out1)
        out2 = out2.reshape(1, -1)
        out3 = np.matmul(self.weights[1], out2.T)
        out3[out3 < 0] = 0  # relu
        out4 = softmax(self.weights[2].dot(out3))
        return np.argmax(out4)

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.weights = pickle.load(fp)
