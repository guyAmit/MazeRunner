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
                                        update_maze, end_near_indicitor)
from models.agent_model import Agent_Model

mazes = [make_maze_from_file(i) for i in range(TRAINSET_SIZE)]
model = Agent_Model(net_type='dense', img_size=MAZE_SIZE[0])
mazes_weights = [1]*TRAINSET_SIZE
mazes_weights[0] = 2
mazes_weights[6] = 2
mazes_weights[7] = 2
mazes_weights[11] = 2
mazes_weights[12] = 2
mazes_weights[17] = 2
mazes_weights[21] = 2


def neural_network_predict(
        model,
        directions_features,
        lstm_featuers,
        current_maze,
        visited,
        iligal_move,
        dead_end,
        end_near):
    if model.net_type == 'cnn':
        pred = model.predict(
            lstm_featuers=directions_features[:4],
            img=current_maze.reshape(
                (1, MAZE_SIZE[0], MAZE_SIZE[1], 1)),
            iligal_move=np.array([iligal_move]),
            dead_end=np.array([dead_end]))

    elif model.net_type == 'lstm':
        lstm_featuers[:, i] = directions_features
        if i != 0:
            lstm_featuers[:, i-1, pred] = -1
        pred = model.predict(lstm_features=lstm_featuers)
    else:
        #  dense
        pred = model.predict(lstm_featuers=directions_features,
                             visited=visited,
                             dead_end=dead_end,
                             iligal_move=1 if iligal_move > 0 else 0,
                             end_near_indicator=end_near)
    return pred


def run_maze(model, maze, debug):
    current_maze = maze[0]
    full_maze = maze[1]
    curr_pos = np.array(STRARTING_POSINGTION)
    prev_pos = curr_pos.copy()
    score = 0
    iligal_move = 0
    dead_end = 0
    lstm_features = np.zeros((1, MAX_STEPS, 6))
    for i in range(MAX_STEPS):
        #  extract maze features:
        directions_features = get_lsm_features(
            current_maze, curr_pos).reshape(-1)
        dead_end = 1 if is_surrounded(
            current_maze, curr_pos) is not None else 0
        visited = (current_maze[curr_pos[0],
                                curr_pos[1]]+1
                   - VISITED_POS if current_maze[curr_pos[0],
                                                 curr_pos[1]]
                   >= VISITED_POS else 0)
        end_near = end_near_indicitor(current_maze, curr_pos)
        pred = neural_network_predict(model, directions_features,
                                      lstm_features,
                                      current_maze, visited, iligal_move,
                                      dead_end, end_near)
        iligal_move = 0
        if (curr_pos[0] + pred[0] >= current_maze.shape[0] or
            curr_pos[1] + pred[1] >= current_maze.shape[1] or
                curr_pos[0] + pred[0] < 0 or curr_pos[1] + pred[1] < 0):
            score += 7  # out of maze
            score += iligal_move
            iligal_move += 1
            # return score
        elif current_maze[curr_pos[0] + pred[0], curr_pos[1]+pred[1]] == END:
            # maze ending bonus
            return score-10*i, 1
        elif current_maze[curr_pos[0] + pred[0], curr_pos[1]+pred[1]] == WALL:
            score += 7  # run into wall
            score += iligal_move
            iligal_move += 1
            # return score
        elif current_maze[curr_pos[0] + pred[0],
                          curr_pos[1]+pred[1]] >= VISITED_POS:
            if model.net_type == 'dense':
                current_maze[curr_pos[0] + pred[0],
                             curr_pos[1]+pred[1]] += 1
            score += current_maze[curr_pos[0] + pred[0],
                                  curr_pos[1]+pred[1]] - 6
            prev_pos = curr_pos.copy()
            curr_pos[0] += pred[0]
            curr_pos[1] += pred[1]
        else:
            score += 1
            prev_pos = curr_pos.copy()
            curr_pos[0] += pred[0]
            curr_pos[1] += pred[1]

        new_tiels = update_maze(current_maze, full_maze, new_pos=curr_pos,
                                old_pos=prev_pos)
        if new_tiels > 0:
            score -= 20
        if debug is True:
            plt.matshow(current_maze)
            plt.show()
    del maze
    return score, 0


def get_reward(weights):
    global model
    global mazes
    global mazes_weights
    model.set_weights(weights)
    reward = 0
    solved = []
    for i, maze in enumerate(mazes):
        r, s = run_maze(model, [maze[0].copy(), maze[1]], debug=False)
        reward += r*mazes_weights[i]
        if s == 1:
            solved.append(i)
    print(f'Reward: {-reward/len(mazes)} Solved: {solved}')
    return -(reward/len(mazes))


if __name__ == '__main__':
    model.load()
    weights = model.get_weights()
    es = EvolutionStrategy(weights, get_reward,
                           population_size=50, sigma=0.15,
                           learning_rate=0.03, num_threads=8)
    for i in range(10):
        print(f'Round number: {i*5}')
        es.run(iterations=5, print_step=1)
        model.save()
