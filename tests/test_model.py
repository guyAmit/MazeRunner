import numpy as np

from maze_runner.models.agent_model import Agent_Model
from maze_runner.consts import MAX_STEPS


def test_output():
    img = np.random.randn(1, 28, 28, 1)
    model = Agent_Model(net_type='cnn', img_size=28)
    pred = model.predict(img, np.array([1]), np.array([1]))
    assert isinstance(pred[0], int) and isinstance(pred[1], int)


def test_set_and_get_weight_cnn():
    model = Agent_Model(net_type='cnn', img_size=30)
    img = np.random.rand(1, 30, 30, 1)
    pred1 = model.predict(img, np.array([1]), np.array([1]))
    weights = model.get_weights()
    model.set_weights(weights)
    pred2 = model.predict(img, np.array([1]), np.array([1]))
    assert (pred1[0] == pred2[0]) and (pred1[1] == pred2[1])


def test_set_and_get_weight_lstm():
    model = Agent_Model(net_type='lstm', img_size=None)
    test = np.random.rand(1, MAX_STEPS, 6)
    pred1 = model.predict(test)
    weights = model.get_weights()
    model.set_weights(weights)
    pred2 = model.predict(test)
    assert (pred1[0] == pred2[0]) and (pred1[1] == pred2[1])
