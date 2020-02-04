import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, cma, creator, tools
from evostra import EvolutionStrategy
import multiprocessing

from consts import MAX_STEPS, MAZE_SIZE, TESTSET_SIZE, TRAINSET_SIZE
from mazes_creator.maze_consts import (END, MAZE_ENDING, STRARTING_POSINGTION,
                                       USER_POS, VISITED_POS, WALL)
from mazes_creator.maze_manager import (end_near_indicitor, get_lsm_features,
                                        is_surrounded, make_maze_from_file,
                                        show_maze, update_maze)
from models.agent_model import Agent_Model

mazes = [make_maze_from_file(i) for i in range(TRAINSET_SIZE)]
model = Agent_Model(algorithm_type='cmaES', img_size=MAZE_SIZE[0])


def get_oposite_direction(i):
    feature = np.zeros((4,))
    if i == 0:
        feature[1] = 1
        return feature
    if i == 1:
        feature[0] = 1
    if i == 2:
        feature[3] = 1
    if i == 3:
        feature[2] = 1
    return feature


def convert_to_directions(pred):
    if pred == 0:
        return (1, 0)
    if pred == 1:
        return (-1, 0)
    if pred == 2:
        return (0, -1)
    if pred == 3:
        return (0, 1)
    return


def times_visited(memory, open_directions):
    feature = np.zeros((4, ))
    min_dir = [MAX_STEPS, 4]
    for i, direction in enumerate(open_directions):
        if direction == 1:
            _dir = convert_to_directions(i)
            visited = memory[_dir[0], _dir[1]]
            if min_dir[0] > visited:
                min_dir = (visited, i)
    feature[min_dir[1]] = 1
    return feature


def neural_network_predict(
        model,
        directions_features,
        oposite_direction,
        current_maze,
        iligal_move,
        end_near,
        times_visited):
    pred = model.predict(lstm_featuers=directions_features,
                         oposite_direction=oposite_direction,
                         end_near_indicator=end_near,
                         times_visited=times_visited)
    return pred


def run_maze(model, maze, debug):
    current_maze = maze[0]
    full_maze = maze[1]
    curr_pos = np.array(STRARTING_POSINGTION)
    prev_pos = curr_pos.copy()
    score = 0
    curr_dirr = 1
    iligal_move = 0
    memeory = np.zeros(full_maze.shape)
    memeory[0, 0] = 1

    for i in range(MAX_STEPS):
        #  extract maze features:
        directions_features = get_lsm_features(
            current_maze, curr_pos).reshape(-1)[:4]
        times_visited_feature = times_visited(memeory, directions_features)
        end_near = end_near_indicitor(current_maze, curr_pos)
        oposite_direction = get_oposite_direction(curr_dirr)
        pred = neural_network_predict(model=model,
                                      directions_features=directions_features,
                                      oposite_direction=oposite_direction,
                                      current_maze=current_maze,
                                      iligal_move=iligal_move,
                                      end_near=end_near,
                                      times_visited=times_visited_feature)
        curr_dirr = pred
        pred = convert_to_directions(pred)
        iligal_move = 0
        if (curr_pos[0] + pred[0] >= current_maze.shape[0] or
            curr_pos[1] + pred[1] >= current_maze.shape[1] or
                curr_pos[0] + pred[0] < 0 or curr_pos[1] + pred[1] < 0):
            score += 5  # out of maze
            score += iligal_move
            iligal_move += 1
            # return score
        elif current_maze[curr_pos[0] + pred[0], curr_pos[1]+pred[1]] == END:
            # maze ending bonus
            return score-3*i, 1
        elif current_maze[curr_pos[0] + pred[0], curr_pos[1]+pred[1]] == WALL:
            score += 5  # run into wall
            score += iligal_move
            iligal_move += 1
            # return score
        elif current_maze[curr_pos[0] + pred[0],
                          curr_pos[1]+pred[1]] >= VISITED_POS:
            score += memeory[curr_pos[0] + pred[0],
                             curr_pos[1]+pred[1]]+1
            prev_pos = curr_pos.copy()
            curr_pos[0] += pred[0]
            curr_pos[1] += pred[1]
        else:
            score -= 2
            prev_pos = curr_pos.copy()
            curr_pos[0] += pred[0]
            curr_pos[1] += pred[1]

        memeory[curr_pos[0], curr_pos[1]] += 1
        new_tiels = update_maze(current_maze, full_maze, new_pos=curr_pos,
                                old_pos=prev_pos)
        if new_tiels > 0:
            score -= 5
        if debug is True:
            plt.matshow(current_maze)
            plt.show()
    del maze
    return score, 0


def get_reward(weights):
    global model
    global mazes
    model.set_weights(weights)
    reward = 0
    solved = []
    for i, maze in enumerate(mazes):
        r, s = run_maze(model, [maze[0].copy(), maze[1]], debug=False)
        reward += r
        if s == 1:
            solved.append(i)
    # print(f'Reward: {-reward/len(mazes)} Solved: {solved}')
    reward = [-reward/len(mazes)]
    return reward if model.algorithm_type == 'OpenAi' else np.array(reward)


def debug_run(weights):
    global model
    global mazes
    model.set_weights(weights)
    reward = 0
    solved = []
    for i, maze in enumerate(mazes):
        r, s = run_maze(model, [maze[0].copy(), maze[1]], debug=False)
        reward += r
        if s == 1:
            solved.append(i)
    print(f'Reward: {-reward/len(mazes)} Solved: {solved}')


if __name__ == '__main__':
    weights = model.get_weights()
    model.load()
    if model.algorithm_type == 'OpenAi':
        # model.load()
        es = EvolutionStrategy(weights, get_reward,
                               population_size=60, sigma=5,
                               learning_rate=0.1, num_threads=8)
        for i in range(20):
            print(f'Round number: {i*5}')
            es.run(iterations=10, print_step=1)
            model.save()

    if model.algorithm_type == 'cmaES':
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("evaluate", get_reward)
        # np.random.seed(128)

        strategy = cma.Strategy(centroid=weights.tolist(),
                                sigma=5.0)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        iterations = 200
        log = []
        for i in range(iterations):
            _, logbook = algorithms.eaGenerateUpdate(
                toolbox, ngen=10, stats=stats, halloffame=hof)
            log = log + logbook
            if (i+1) % 3 == 0:
                weights = hof[0]
                debug_run(weights)
                MAX_STEPS += 5
            model.set_weights(hof[0])
            model.save()
            pickle.dump(log, open('stats.pkl', 'wb'))
    # run_maze(model, mazes[0], debug=True)
