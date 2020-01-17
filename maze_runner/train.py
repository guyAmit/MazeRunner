from evostra import EvolutionStrategy
from models.model import Model
from mazes_creator.maze_manager import (
    make_maze_from_file, show_maze, update_maze)
from mazes_creator.maze_consts import (
    STRARTING_POSINGTION, WALL, MAZE_ENDING, USER_POS, END)
from consts import TESTSET_SIZE, TRAINSET_SIZE, MAZE_SIZE, MAX_STEPS
import numpy as np
import matplotlib.pyplot as plt


def run_maze(model, maze):
    current_maze = maze[0]
    current_maze[STRARTING_POSINGTION[0], STRARTING_POSINGTION[1]] = USER_POS
    full_maze = maze[1]
    curr_pos = np.array(STRARTING_POSINGTION)
    prev_pos = curr_pos.copy()
    score = 0
    for i in range(MAX_STEPS):
        pred = model.predict(current_maze)
        if current_maze[curr_pos[0] + pred[0], curr_pos[1]+pred[1]] == END:
        # if (curr_pos[0] + pred[0] == MAZE_ENDING[0] and
        #         curr_pos[1]+pred[1] == MAZE_ENDING[1]):
            score -= 4  # maze ending bonus
            return score
        elif current_maze[curr_pos[0] + pred[0], curr_pos[1]+pred[1]] == WALL:
            score += 2  # run into wall
        elif (curr_pos[0] + pred[0] >= current_maze.shape[0] or
              curr_pos[1] + pred[1] >= current_maze.shape[1] or
                curr_pos[0] + pred[0] < 0 or curr_pos[1] + pred[1] < 0):
            score += 4  # out of maze
        else:
            score += 1
            prev_pos = curr_pos.copy()
            curr_pos[0] += pred[0]
            curr_pos[1] += pred[1]

        update_maze(current_maze, full_maze, new_pos=curr_pos,
                    old_pos=prev_pos)
        # a = np.array(list(current_maze))
        # plt.imshow(-a)
        # plt.show()
    return score


def reward_func(mazes, model):
    def get_reward(weights):
        model.set_weights(weights)
        reward = 0
        for maze in mazes:
            reward += run_maze(model, maze)
        return -1*(reward / len(mazes))
    return get_reward


if __name__ == '__main__':
    mazes = [make_maze_from_file(i) for i in range(TRAINSET_SIZE)]
    model = Model(fillters_number=10, dense_size=10, img_size=MAZE_SIZE[0])
    weights = model.get_weights()
    es = EvolutionStrategy(weights, reward_func(mazes, model),
                           population_size=100, sigma=0.2,
                           learning_rate=0.3, num_threads=1)

    es.run(iterations=10, print_step=1)
    model.save()
