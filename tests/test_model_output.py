import numpy as np

from maze_runner.models.model import Model


def test_model_output():
    img = np.random.randn(28, 28)
    model = Model(fillters_number=10, dense_size=10, img_size=28)
    assert isinstance(model.predict(img), np.int32)
