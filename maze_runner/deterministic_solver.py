import random
import matplotlib.pyplot as plt
import numpy as np

from consts import MAX_STEPS, MAZE_SIZE, TESTSET_SIZE, TRAINSET_SIZE
from mazes_creator.maze_consts import (END, MAZE_ENDING, STRARTING_POSINGTION,
                                       USER_POS, VISITED_POS, WALL)
from mazes_creator.maze_manager import (end_near_indicitor, get_lsm_features,
                                        is_surrounded, make_maze_from_file,
                                        show_maze, update_maze)

mazes = [make_maze_from_file(i) for i in range(TRAINSET_SIZE)]


def predict(curr_pos, curr_direction,
            current_maze, directions_features, end_near, memeory):
    # print(directions_features)
    open_directoins = []
    for i, x in enumerate(zip(directions_features, end_near)):
        if x[1] == 1:
            return i
        if x[0] == 1:
            open_directoins.append(i)
    if len(open_directoins) > 1:
        visited = []
        times_visited_min = (MAX_STEPS, MAX_STEPS)
        for x in open_directoins:
            direction = convert_to_directions(x)
            times_visited = memeory[curr_pos[0]+direction[0],
                                    curr_pos[1]+direction[1]]
            if times_visited >= 1:
                visited.append(x)
                if (times_visited_min[0] > times_visited and
                        x != get_oposite_direction(curr_direction)):
                    times_visited_min = (times_visited, x)
            else:
                return x
        if len(visited) >= 2:
            return times_visited_min[1]
        else:
            return visited[0]

    return open_directoins[0]


def run_maze(maze, debug):
    current_maze = maze[0]
    full_maze = maze[1]
    curr_pos = np.array(STRARTING_POSINGTION)
    prev_pos = curr_pos.copy()
    score = 0
    curr_dirr = 1
    memeory = np.zeros(full_maze.shape)
    memeory[0, 0] = 1
    for i in range(MAX_STEPS):
        #  extract maze features:
        directions_features = get_lsm_features(
            current_maze, curr_pos).reshape(-1)[:4]
        end_near = end_near_indicitor(current_maze, curr_pos)
        pred = predict(curr_pos, curr_dirr, current_maze,
                       directions_features, end_near, memeory)
        curr_dirr = pred
        pred = convert_to_directions(pred)
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
        memeory[curr_pos[0], curr_pos[1]] += 1
        new_tiels = update_maze(current_maze, full_maze, new_pos=curr_pos,
                                old_pos=prev_pos)
        if new_tiels > 0:
            score -= 20
        if debug is True:
            plt.matshow(current_maze)
            plt.show()
    del maze, memeory
    return score, 0


def get_reward():
    reward = 0
    solved = []
    for i, maze in enumerate(mazes):
        r, s = run_maze([maze[0].copy(), maze[1]], debug=False)
        reward += r
        if s == 1:
            solved.append(i)
    print(f'Reward: {-reward/len(mazes)} Solved: {solved}')
    return -(reward/len(mazes))


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


def get_oposite_direction(i):
    if i == 0:
        return 1
    if i == 1:
        return 0
    if i == 2:
        return 3
    if i == 3:
        return 2
    return


if __name__ == '__main__':
    # run_maze(mazes[0], debug=False)
    print(get_reward())
