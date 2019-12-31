# import pickle

import numpy as np
# import scipy.signal

from maze_runner.models.model import Model


def test_foo():
    img = np.random.randn(28, 28)
    model = Model(fillters_number=10, dense_size=10, img_size=28)
    print(model.predict(img))
