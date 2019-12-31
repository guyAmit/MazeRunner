import numpy as np

from maze_runner.models.model import Model


<<<<<<< HEAD:tests/test_model.py
def test_output():
=======
def test_model_output():
>>>>>>> c6dfd9ba832cd42684e1b9544002fac91d14f731:tests/test_model_output.py
    img = np.random.randn(28, 28)
    model = Model(fillters_number=10, dense_size=10, img_size=28)
    assert isinstance(model.predict(img), np.int32)
