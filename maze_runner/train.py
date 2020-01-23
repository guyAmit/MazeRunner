import datetime

import matplotlib.pyplot as plt
import numpy as np
from evostra import EvolutionStrategy

from consts import MAX_STEPS, MAZE_SIZE, TESTSET_SIZE, TRAINSET_SIZE
from gif_maker import make_gif
from mazes_creator.maze_consts import (END, MAZE_ENDING, STRARTING_POSINGTION,
                                       USER_POS, VISITED_POS, WALL)
from mazes_creator.maze_manager import (get_lsm_features, is_surrounded,
                                        make_maze_from_file, show_maze,
                                        update_maze)
from models.agent_model import Agent_Model

global SOLVED
SOLVED = set()


def convert_array(current_maze):
    l = list(current_maze)
    np_array = -np.array(l)

    # res = np_array.astype(np.int8)
    return np_array


def run_maze(model, maze, j):
    global SOLVED
    current_maze = maze[0]
    full_maze = maze[1]
    curr_pos = np.array(STRARTING_POSINGTION)
    prev_pos = curr_pos.copy()
    score = 0
    iligal_move = 0
    dead_end = 0
    features = np.zeros((1, MAX_STEPS, 6))
    mazes = []
    for i in range(MAX_STEPS):
        directions_features = get_lsm_features(current_maze, curr_pos)
        if model.net_type == 'cnn':
            dead_end = 1 if is_surrounded(
                current_maze, curr_pos) is not None else 0

            pred = model.predict(
                lstm_featuers=directions_features[:4],
                img=current_maze.reshape(
                    (1, MAZE_SIZE[0], MAZE_SIZE[1], 1)),
                iligal_move=np.array([iligal_move]),
                dead_end=np.array([dead_end]))

        if model.net_type == 'lstm':
            features[:, i] = directions_features
            if i != 0:
                features[:, i-1, pred] = -1
            pred = model.predict(lstm_features=features)
        iligal_move = 0
        if (curr_pos[0] + pred[0] >= current_maze.shape[0] or
            curr_pos[1] + pred[1] >= current_maze.shape[1] or
                curr_pos[0] + pred[0] < 0 or curr_pos[1] + pred[1] < 0):
            score += 2.5  # out of maze
            iligal_move = 1
            # return score
        elif current_maze[curr_pos[0] + pred[0], curr_pos[1]+pred[1]] == END:
            score -= 1000  # maze ending bonus
            print('finished maze !!')
            if j not in SOLVED:
                make_gif(mazes, j)
                SOLVED.add(j)
                return score+i*3
            return score+i*10
        elif current_maze[curr_pos[0] + pred[0], curr_pos[1]+pred[1]] == WALL:
            score += 2  # run into wall
            iligal_move = 1
            # return score
        elif current_maze[curr_pos[0] + pred[0],
                          curr_pos[1]+pred[1]] == VISITED_POS:
            score -= 2
            prev_pos = curr_pos.copy()
            curr_pos[0] += pred[0]
            curr_pos[1] += pred[1]
        else:
            score -= 5
            prev_pos = curr_pos.copy()
            curr_pos[0] += pred[0]
            curr_pos[1] += pred[1]

        new_tiels = update_maze(current_maze, full_maze, new_pos=curr_pos,
                                old_pos=prev_pos)
        converted_maze = convert_array(current_maze)
        mazes.append(converted_maze)
        if new_tiels > 0:
            score -= 15

    del maze
    return score


def reward_func(mazes, model):
    def get_reward(weights):
        model.set_weights(weights)
        reward = 0
        counter = 0
        for maze in mazes:
            reward += run_maze(model, [maze[0].copy(), maze[1]], counter)
            counter += 1
        print(reward)
        return -(reward)
    return get_reward


if __name__ == '__main__':
    mazes = [make_maze_from_file(i) for i in range(TRAINSET_SIZE)]
    model = Agent_Model(net_type='cnn', img_size=MAZE_SIZE[0])
    # model.load()
    weights = model.get_weights()
    es = EvolutionStrategy(weights, reward_func(mazes, model),
                           population_size=60, sigma=0.15,
                           learning_rate=0.05, num_threads=1)

    es.run(iterations=1000, print_step=1)
    model.save()
