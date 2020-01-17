import matplotlib.pyplot as plt
import numpy as np
# from daedalus import Maze
# from daedalus._maze import init_random

# from maze_consts import

from mazes_creator.maze_consts import WALL, USER_POS, OPEN, UNSEEN, END


def _get_maze_at_pos(maze: np.array, pos: (int, int)):
    return maze[pos[0]][pos[1]]


def _set_maze_at_post(maze: np.array, pos: (int, int), val: int):
    if pos[1] >= len(maze) or pos[0] >= len(maze):
        return maze
    maze[pos[0]][pos[1]] = val
    return maze


def _revel_in_1_pos(new_maze, old_maze, pos):
    val = _get_maze_at_pos(old_maze, pos)
    maze = _set_maze_at_post(new_maze, pos, val)
    return maze


def _revel_in_pos(new_maze, old_maze, pos):
    val = _get_maze_at_pos(old_maze, pos)
    maze = _set_maze_at_post(new_maze, pos, val)
    if _get_maze_at_pos(old_maze, pos) == WALL:
        _revel_in_1_pos(new_maze, old_maze, pos)
        return new_maze
    if pos[0] > 0:
        _revel_in_1_pos(new_maze, old_maze, [pos[0]-1, pos[1]])
    if pos[1] > 0:
        _revel_in_1_pos(new_maze, old_maze, [pos[0], pos[1]-1])
    if pos[0] < len(new_maze)-1:
        _revel_in_1_pos(new_maze, old_maze, [pos[0]+1, pos[1]])
    if pos[1] < len(new_maze)-1:
        _revel_in_1_pos(new_maze, old_maze, [pos[0], pos[1]+1])
    return maze


def look_left(maze: np.array, full_maze: np.array, pos: (int, int)):
    current_pos = np.copy(pos)
    current_pos[0] -= 1
    try:
        current_val = _get_maze_at_pos(maze, current_pos)
    except:
        return maze
    while current_val == UNSEEN or current_val == OPEN:
        if current_pos[0] < 0:
            break
        if _get_maze_at_pos(full_maze, current_pos) == WALL:
            maze = _revel_in_1_pos(maze, full_maze, current_pos)
            break
        maze = _revel_in_pos(maze, full_maze, current_pos)
        current_pos[0] -= 1
    return maze


def look_right(maze: np.array, full_maze: np.array, pos: (int, int)):
    current_pos = np.copy(pos)
    current_pos[0] += 1
    try:
        current_val = _get_maze_at_pos(maze, current_pos)
    except:
        return maze
    while current_val == UNSEEN or current_val == OPEN:
        if current_pos[0] >= len(maze):
            break
        if _get_maze_at_pos(full_maze, current_pos) == WALL:
            maze = _revel_in_1_pos(maze, full_maze, current_pos)
            break
        maze = _revel_in_pos(maze, full_maze, current_pos)
        current_pos[0] += 1
    return maze


def look_up(maze: np.array, full_maze: np.array, pos: [int, int]):
    current_pos = np.copy(pos)
    current_pos[1] -= 1  # go one down
    try:
        current_val = _get_maze_at_pos(maze, current_pos)
    except:
        return maze
    while current_val == UNSEEN or current_val == OPEN:
        if current_pos[1] < 0:
            break
        if _get_maze_at_pos(full_maze, current_pos) == WALL:
            maze = _revel_in_1_pos(maze, full_maze, current_pos)
            break
        maze = _revel_in_pos(maze, full_maze, current_pos)
        current_pos[1] -= 1
    return maze


def look_down(maze: np.array, full_maze: np.array, pos: [int, int]):
    current_pos = np.copy(pos)
    current_pos[1] += 1  # go one down
    try:
        current_val = _get_maze_at_pos(maze, current_pos)
    except:
        return maze
    while current_val == UNSEEN or current_val == OPEN:
        if current_pos[1] >= len(maze):
            break
        if _get_maze_at_pos(full_maze, current_pos) == WALL:
            maze = _revel_in_1_pos(maze, full_maze, current_pos)
            break
        maze = _revel_in_pos(maze, full_maze, current_pos)
        current_pos[1] += 1
    return maze


def update_maze(current_maze: np.array, full_maze: np.array,
                new_pos: [int, int], old_pos: [int, int]):
    # current_maze = _set_maze_at_post(current_maze, old_pos, _get_maze_at_pos(full_maze, old_pos))
    _revel_in_pos(current_maze, full_maze, old_pos)
    # current_maze[old_pos[0]][old_pos[1]]=_get_maze_at_pos(full_maze,old_pos)
    current_maze = _set_maze_at_post(current_maze, new_pos, USER_POS)
    look_down(current_maze, full_maze, new_pos)
    look_right(current_maze, full_maze, new_pos)
    look_left(current_maze, full_maze, new_pos)
    look_up(current_maze, full_maze, new_pos)
    current_maze[new_pos[0]][new_pos[1]] = USER_POS


"""
1 wall
0 open
-1 unseen
999 user
"""


def make_maze(size, seed):
    real_maze = Maze(*size)
    init_random(seed)
    Maze.create_perfect2(real_maze, nEntrancePos=0)
    known_maze = np.ndarray(shape=(size[0], size[1]), dtype=int)
    known_maze.fill(UNSEEN)
    # m = Maze.create_perfect(maze, nEntrancePos=0, nRndBias=2)
    full_maze = np.array(list(real_maze))
    full_maze[0, 0] = OPEN
    full_maze[real_maze.exit[0], real_maze.exit[1]] = END
    update_maze(known_maze, full_maze,
                [0, 0],
                [0, 0])
    # makr_entrance(arr, real_maze.entrance)
    return known_maze, full_maze


def make_maze_from_file(index):
    m = np.load('mazes_creator//mazes.npy')
    known, full = m[index]
    return known, full


def show_maze(maze):
    print('Maze entrance: {}'.format(maze.entrance))
    print('Maze exit: {}'.format(maze.exit))
    a = np.array(list(maze))
    plt.imshow(-a)
    plt.show()
    pass


# def main():
#     SIZE = 31
#     m = make_maze((SIZE, SIZE), 1)
#     show_maze(m)


if __name__ == '__main__':
    mazes = []
    # m = np.load('mazes.npy')
    pass
    for i in range(100):
        known, full = make_maze((30, 30), i)
        # np.save
        # np.save(f'{i}_{known}.npy',)
        mazes.append((known, full))

    np.save('mazes', mazes)
