from evostra import EvolutionStrategy
from models.agent_model import Agent_Model
from mazes_creator.maze_manager import (
    make_maze_from_file, show_maze, update_maze, is_surrounded)
from mazes_creator.maze_consts import (
    STRARTING_POSINGTION, WALL, MAZE_ENDING, USER_POS, END)
from consts import TESTSET_SIZE, TRAINSET_SIZE, MAZE_SIZE, MAX_STEPS
import numpy as np
import matplotlib.pyplot as plt


def run_maze(model, maze):
    current_maze = maze[0]
    # current_maze[STRARTING_POSINGTION[0], STRARTING_POSINGTION[1]] = USER_POS
    full_maze = maze[1]
    curr_pos = np.array(STRARTING_POSINGTION)
    prev_pos = curr_pos.copy()
    score = 0
    iligal_move = 0
    dead_end = 0
    for i in range(MAX_STEPS):
        dead_end = 1 if is_surrounded(
            current_maze, curr_pos) is not None else 0
        pred = model.predict(current_maze.reshape(
            1, MAZE_SIZE[0], MAZE_SIZE[1], 1),
            np.array([iligal_move]), np.array([dead_end]))

        iligal_move = 0
        if current_maze[curr_pos[0] + pred[0], curr_pos[1]+pred[1]] == END:
            score -= 20  # maze ending bonus
            print('finished maze !!')
            return score
        elif current_maze[curr_pos[0] + pred[0], curr_pos[1]+pred[1]] == WALL:
            score += 2  # run into wall
            iligal_move = 1

        elif (curr_pos[0] + pred[0] >= current_maze.shape[0] or
              curr_pos[1] + pred[1] >= current_maze.shape[1] or
                curr_pos[0] + pred[0] < 0 or curr_pos[1] + pred[1] < 0):
            score += 2.5  # out of maze
            iligal_move = 1

        else:
            score += 1
            prev_pos = curr_pos.copy()
            curr_pos[0] += pred[0]
            curr_pos[1] += pred[1]

        new_tiels = update_maze(current_maze, full_maze, new_pos=curr_pos,
                                old_pos=prev_pos)
        if new_tiels > 0:
            score -= 0.5
        # plt.matshow(current_maze)
        # plt.show()
    del maze
    return score


def reward_func(mazes, model):
    def get_reward(weights):
        model.set_weights(weights)
        reward = 0
        for maze in mazes:
            reward += run_maze(model, [maze[0].copy(), maze[1]])
        print(reward)
        return -1*(reward / len(mazes))
    return get_reward


if __name__ == '__main__':
    mazes = [make_maze_from_file(i) for i in range(TRAINSET_SIZE)]
    model = Agent_Model(img_size=MAZE_SIZE[0])
    # model.load()
    weights = model.get_weights()
    es = EvolutionStrategy(weights, reward_func(mazes, model),
                           population_size=300, sigma=0.22,
                           learning_rate=0.01, num_threads=1)

    es.run(iterations=1000, print_step=1)
    model.save()
