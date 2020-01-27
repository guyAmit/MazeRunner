import math
from copy import copy
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from consts import UP, DOWN, LEFT, RIGHT
from .maze_consts import WALL, USER_POS, OPEN, UNSEEN, END, VISITED_POS


from daedalus import Maze
from daedalus._maze import init_random
# from maze_consts import WALL, USER_POS, OPEN, UNSEEN, END, VISITED_POS

class MazeManager():
    def __init__(self, full_maze,known_maze):
        self.full_maze = full_maze
        self.known_maze = known_maze
        self.current_pos = [0,0]
        self.prev_pos= [0,0]
    def _get_maze_at_pos(self, pos: (int, int)):
        try:
            res = self.full_maze[pos[0]][pos[1]]
            return res
        except IndexError:
            return None


    def _set_maze_at_post(self, pos: (int, int), val: int):
        if pos[1] >= len(self.full_maze) or pos[0] >= len(self.full_maze):
            return self.full_maze
        if self._get_maze_at_pos(pos) == VISITED_POS:
            return self.full_maze
        self.full_maze[pos[0]][pos[1]] = val
        return self.full_maze


    def _revel_in_1_pos(self, pos):
        res = 0
        if self._get_maze_at_pos(pos) == UNSEEN:
            res = 1
        val = self._get_maze_at_pos(pos)
        self._set_maze_at_post(pos, val)
        return res

    def get_dir_closest_to_end(self):
        cur = copy(self.current_pos)
        left =self.get_left(cur)
        right=self.get_right(cur)
        up=self.get_up(cur)
        down=self.get_down(cur)
        dirs ={LEFT:left,RIGHT:right,UP:up,DOWN:down}
        max_move=(LEFT,99999)
        for dir, p in dirs.items():
            if self._is_valid_move(p):
                dist = self.dist_from_end(p)
                if dist<max_move[1]:
                    max_move=(dir,dist)
        return max_move[0]


    def _revel_in_pos(self, pos):
        val = self._get_maze_at_pos(pos)
        self._set_maze_at_post(pos, val)
        updated_count = 0
        if self._get_maze_at_pos(pos) == WALL:
            return self._revel_in_1_pos(pos)
        if pos[0] > 0:
            updated_count += self._revel_in_1_pos([pos[0] - 1, pos[1]])
        if pos[1] > 0:
            updated_count += self._revel_in_1_pos([pos[0], pos[1] - 1])
        if pos[0] < len(self.known_maze) - 1:
            updated_count += self._revel_in_1_pos([pos[0] + 1, pos[1]])
        if pos[1] < len(self.known_maze) - 1:
            updated_count += self._revel_in_1_pos([pos[0], pos[1] + 1])
        return updated_count


    def look_left(self, pos: (int, int)):
        current_pos = np.copy(pos)
        current_pos[0] -= 1
        updated_count = 0
        try:
            current_val = self._get_maze_at_pos(current_pos)
        except ValueError:
            return 0
        while current_val == UNSEEN or current_val == OPEN:
            if current_pos[0] < 0:
                break
            if self._get_maze_at_pos(current_pos) == WALL:
                updated_count += self._revel_in_1_pos(current_pos)
                break
            updated_count += self._revel_in_pos(current_pos)
            current_pos[0] -= 1
        return updated_count


    def look_right(self, pos: (int, int)):
        current_pos = np.copy(pos)
        current_pos[0] += 1
        updated_count = 0
        try:
            current_val = self._get_maze_at_pos(current_pos)
        except ValueError:
            return 0
        while current_val == UNSEEN or current_val == OPEN:
            if current_pos[0] >= len(self.full_maze):
                break
            if self._get_maze_at_pos(current_pos) == WALL:
                updated_count += self._revel_in_1_pos(current_pos)
                break
            updated_count += self._revel_in_pos(current_pos)
            current_pos[0] += 1
        return updated_count


    def look_up(self, pos: [int, int]):
        current_pos = np.copy(pos)
        current_pos[1] -= 1  # go one down
        updated_count = 0
        try:
            current_val = self._get_maze_at_pos(current_pos)
        except ValueError:
            return 0
        while current_val == UNSEEN or current_val == OPEN:
            if current_pos[1] < 0:
                break
            if self._get_maze_at_pos(current_pos) == WALL:
                updated_count += self._revel_in_1_pos(current_pos)
                break
            updated_count += self._revel_in_pos(current_pos)
            current_pos[1] -= 1
        return updated_count


    def look_down(self, pos: [int, int]):
        current_pos = np.copy(pos)
        current_pos[1] += 1  # go one down
        updated_count = 0
        try:
            current_val = self._get_maze_at_pos(current_pos)
        except ValueError:
            return 0
        while current_val == UNSEEN or current_val == OPEN:
            if current_pos[1] >= len(self.full_maze):
                break
            if self._get_maze_at_pos(current_pos) == WALL:
                updated_count += self._revel_in_1_pos(current_pos)
                break
            updated_count += self._revel_in_pos(current_pos)
            current_pos[1] += 1
        return updated_count


    def is_surrounded(self, pos):
        up = [pos[0] - 1, pos[1]]
        down = [pos[0] + 1, pos[1]]
        left = [pos[0], pos[1] - 1]
        right = [pos[0], pos[1] + 1]
        try:
            if (self._get_maze_at_pos(up) == WALL and
                    self._get_maze_at_pos(down) == WALL and
                    self._get_maze_at_pos(left) == WALL):
                return (0, 1)
            if (self._get_maze_at_pos(up) == WALL and
                    self._get_maze_at_pos(down) == WALL and
                    self._get_maze_at_pos(right) == WALL):
                return (0, -1)
            if (self._get_maze_at_pos( up) == WALL and
                    self._get_maze_at_pos(left) == WALL and
                    self._get_maze_at_pos( right) == WALL):
                return (1, 0)
            if (self._get_maze_at_pos(left) == WALL and
                    self._get_maze_at_pos(down) == WALL and
                    self._get_maze_at_pos(right) == WALL):
                return (-1, 0)
            return None
        except ValueError:
            return None

    def is_prev_pos_up(self):
        cur_pos = copy(self.current_pos)
        down = self.get_down(cur_pos)
        return self.prev_pos[0]==down[0] and self.prev_pos[1]==down[1]

    def is_prev_pos_down(self):
        cur_pos = copy(self.current_pos)
        up = self.get_up(cur_pos)
        return self.prev_pos[0] == up[0] and self.prev_pos[1] == up[1]

    def is_prev_pos_left(self):
        cur_pos = copy(self.current_pos)
        right = self.get_right(cur_pos)
        return self.prev_pos[0] == right[0] and self.prev_pos[1] == right[1]

    def is_prev_pos_right(self):
        cur_pos = copy(self.current_pos)
        left = self.get_left(cur_pos)
        return self.prev_pos[0] == left[0] and self.prev_pos[1] == left[1]

    def is_dead_end_down(self, pos, res=False):
        val = self._get_maze_at_pos(pos)
        if val is None:
            return res
        if val == WALL:
            return res
        if val is OPEN:
            left_pos = [pos[0] - 1, pos[1]]
            right_pos = [pos[0] + 1, pos[1]]
            left_val = self._get_maze_at_pos(left_pos)
            right_val = self._get_maze_at_pos(right_pos)
            if left_val == OPEN or right_val == OPEN:
                return True
            return self.is_dead_end_down([pos[0] + 1, pos[1]], res)


    def is_dead_end_up(self, pos, res=False):
        val = self._get_maze_at_pos(pos)
        if val is None:
            return res
        if val == WALL:
            return res
        if val is OPEN:
            left_pos = [pos[0] - 1, pos[1]]
            right_pos = [pos[0] + 1, pos[1]]
            left_val = self._get_maze_at_pos(left_pos)
            right_val = self._get_maze_at_pos(right_pos)
            if left_val == OPEN or right_val == OPEN:
                return True
            return self.is_dead_end_down( [pos[0] - 1, pos[1]], res)


    def is_dead_end_right(self, pos, res=False):
        val = self._get_maze_at_pos( pos)
        if val is None:
            return res
        if val == WALL:
            return res
        if val is OPEN:
            down_pos = [pos[0], pos[1] - 1]
            up_pos = [pos[0] + 1, pos[1] + 1]
            down_val = self._get_maze_at_pos(down_pos)
            up_val = self._get_maze_at_pos(up_pos)
            if down_val == OPEN or up_val == OPEN:
                return True
            return self.is_dead_end_down([pos[0], pos[1] + 1], res)


    def is_dead_end_left(self, pos, res=False):
        val = self._get_maze_at_pos(pos)
        if val is None:
            return res
        if val == WALL:
            return res
        if val is OPEN:
            down_pos = [pos[0], pos[1] - 1]
            up_pos = [pos[0] + 1, pos[1] + 1]
            down_val = self._get_maze_at_pos(down_pos)
            up_val = self._get_maze_at_pos(up_pos)
            if down_val == OPEN or up_val == OPEN:
                return True
            return self.is_dead_end_down([pos[0], pos[1] - 1], res)


    def dist_from_end(self, pos):
        maze_size = len(self.full_maze)
        dx = maze_size - 1 - pos[1]
        dy = maze_size - 1 - pos[0]
        dx = dx ^ 2
        dy = dy ^ 2
        res = sqrt(dx + dy)
        return res


    def angle_from_end(self, pos):
        maze_size = len(self.full_maze)
        dx = abs(maze_size - 1 - pos[1])
        dy = abs(maze_size - 1 - pos[0])
        res = math.atan(dy / dx)
        return res

    def get_up(self,pos):
        return [pos[0] + 1, pos[1]]

    def get_down(self,pos):
        return [pos[0] - 1, pos[1]]

    def get_left(self,pos):
        return [pos[0], pos[1] - 1]

    def get_right(self,pos):
        return [pos[0], pos[1] + 1]

    def update_maze(self,
                    new_pos: [int, int], old_pos: [int, int]):
        if new_pos[0]>=len(self.full_maze) or new_pos[1]>=len(self.full_maze) or new_pos[0]<0 or new_pos[1]<0:
            return -0
        if self._get_maze_at_pos(new_pos)==WALL:
            return -1
        updated_count = 0
        # current_maze = _set_maze_at_post(current_maze, old_pos, _get_maze_at_pos(full_maze, old_pos))
        updated_count += self._revel_in_pos( old_pos)
        # current_maze[old_pos[0]][old_pos[1]]=_get_maze_at_pos(full_maze,old_pos)
        current_maze = self._set_maze_at_post( new_pos, USER_POS)
        updated_count += self.look_down( new_pos)
        updated_count += self.look_right(new_pos)
        updated_count += self.look_left( new_pos)
        updated_count += self.look_up(new_pos)
        current_maze[new_pos[0]][new_pos[1]] = USER_POS
        current_maze[old_pos[0]][old_pos[1]] = VISITED_POS
        self.current_pos=new_pos
        self.prev_pos=old_pos
        return updated_count

    def go_up(self):
        new_pos = copy(self.current_pos)
        new_pos = self.get_up(new_pos)
        return self.update_maze(new_pos,self.current_pos)

    def go_down(self):
        new_pos = copy(self.current_pos)
        new_pos = self.get_down(new_pos)
        return self.update_maze(new_pos, self.current_pos)
    def go_left(self):
        new_pos = copy(self.current_pos)
        new_pos = self.get_left(new_pos)
        return self.update_maze(new_pos, self.current_pos)
    def go_right(self):
        new_pos = copy(self.current_pos)
        new_pos = self.get_right(new_pos)
        return self.update_maze(new_pos, self.current_pos)
    def is_finished(self):
        res = self.current_pos[0]+1==len(self.full_maze) and self.current_pos[1]+1==len(self.full_maze)
        return res
    def is_up_wall(self):
        res = self._get_maze_at_pos(self.get_up(self.current_pos))==WALL
        if res is None:
            return True
        return res
    def is_down_wall(self):
        res = self._get_maze_at_pos(self.get_down(self.current_pos)) == WALL
        if res is None:
            return True
        return res
    def is_left_wall(self):
        res = self._get_maze_at_pos(self.get_left(self.current_pos))==WALL
        if res is None:
            return True
        return res
    def is_right_wall(self):
        res = self._get_maze_at_pos(self.get_right(self.current_pos))==WALL
        if res is None:
            return True
        return res
    def get_lsm_features(self, pos):
        directions_vals = [self._get_maze_at_pos(self.get_up(pos)),
                           self._get_maze_at_pos(self.get_down(pos)),
                           self._get_maze_at_pos(self.get_left(pos)),
                           self._get_maze_at_pos(self.get_right(pos))]
        res = []
        for p in directions_vals:
            if p == OPEN:
                res.append(1)
            else:
                res.append(0)
        dist = self.dist_from_end( pos)
        angle = self.angle_from_end(pos)
        res.append(dist)
        res.append(angle)
        res = np.array(res)
        return res
    def did_visit_in_pos(self, pos):
        try:
            val = self._get_maze_at_pos(pos)
            return val==VISITED_POS
        except:
            return False

    def did_visit(self, dir):
        up = [self.current_pos[0] - 1, self.current_pos[1]]
        down = [self.current_pos[0] + 1, self.current_pos[1]]
        left = [self.current_pos[0], self.current_pos[1] - 1]
        right = [self.current_pos[0], self.current_pos[1] + 1]

        if dir ==UP:
            return self.did_visit_in_pos(up)
        if dir==DOWN:
            return self.did_visit_in_pos(down)
        if dir==LEFT:
            return self.did_visit_in_pos(left)
        if dir==RIGHT:
            return self.did_visit_in_pos(right)
    def _is_valid_move(self, new_pos):
        try:
            res = self._get_maze_at_pos(new_pos)
            if res != WALL:
                return True
            return False
        except:
            return False
    def get_valid_moves(self):
        moves =[]
        right_pos = self.get_right(self.current_pos)
        left_pos = self.get_left(self.current_pos)
        up_pos=self.get_up(self.current_pos)
        down_pos=self.get_down(self.current_pos)
        if self._is_valid_move(right_pos):
            moves.append(RIGHT)
        if self._is_valid_move(left_pos):
            moves.append(LEFT)
        if self._is_valid_move(up_pos):
            moves.append(UP)
        if self._is_valid_move(down_pos):
            moves.append(DOWN)
        return moves
    def move(self, direction):
        old_pos = copy(self.current_pos)
        new_pos = None
        if direction == UP:
            new_pos=self.get_up(old_pos)
        if direction == DOWN:
            new_pos=self.get_down(old_pos)
        if direction == LEFT:
            new_pos=self.get_left(old_pos)
        if direction == RIGHT:
            new_pos=self.get_right(old_pos)
        return self.update_maze(new_pos,old_pos)
# def maze_go_left(new_maze, old_maze, curre_pos)

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
    size = size[0]
    full_maze[size-1, size-1] = END
    full_maze[size - 2, size - 1] = OPEN
    full_maze[size - 1, size - 2] = OPEN
    mng =MazeManager(full_maze,known_maze)
    mng.update_maze(
                [0, 0],
                [0, 0])
    # makr_entrance(arr, real_maze.entrance)
    return mng.known_maze, mng.full_maze


def make_maze_from_file(index):
    m = np.load('mazes_creator//mazes.npy')
    known, full = m[index]
    return known, full

def make_maze_manger_from_file(index):
    m = np.load('mazes_creator//mazes.npy')
    known, full = m[index]
    mngr = MazeManager(full,known)
    mngr.update_maze(
                [0, 0],
                [0, 0])
    return mngr







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
        known, full = make_maze((13, 13), i)
        plt.imshow(full)
        plt.show()
        # np.save
        # np.save(f'{i}_{known}.npy',)
        mazes.append((known, full))

    np.save('mazes', mazes)
