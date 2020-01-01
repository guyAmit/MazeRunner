from evostra import EvolutionStrategy
from .models.model import Model
from .mazes_creator import main


def run_maze(model, maze):
    pass


def reward_func(mazes, model):
    def get_reward(weights):
        model.set_weights(weights)
        reward = 0
        for maze in mazes:
            reward += run_maze(model, maze)
        return reward / len(mazes)
    return get_reward


if __name__ == '__main__':
    mazes = main()
    model = Model(fillters_number=10, dense_size=10, img_size=28)
    es = EvolutionStrategy(model.get_weights, reward_func(mazes, model))
    es.run(iterations=100, print_step=10)
