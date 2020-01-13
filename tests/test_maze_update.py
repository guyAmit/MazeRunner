import unittest
from mazes_creator.maze_manager import *

class TestMazeUpdate(unittest.TestCase):
    def _equate_mazes(self, m1,m2):
        return np.array_equal(m1,m2)
    def test_maze_builder_random_seed(self):
        size=30
        maze1 =  make_maze((size,size), 1)
        maze2= make_maze((size,size),1)
        eq1 = self._equate_mazes(maze1[0],maze2[0])
        eq2= self._equate_mazes(maze1[1],maze2[1])
        self.assertTrue(eq1)
        self.assertTrue(eq2)

    def test_maze_update_simple(self):
        size=3
        known_maze, real_maze = make_maze((size,size),1)
        update_maze(known_maze,real_maze,[1,1],[0,0])
        res = self._equate_mazes(known_maze,real_maze)
        self.assertTrue(res)

    def test_unknown_maze(self):
        size=3
        known_maze, real_maze = make_maze((size,size),1)
        res = self._equate_mazes(known_maze,real_maze)
        self.assertFalse(res)
    def test_update_maze_left_right_up_down(self):
        size = 30
        known_maze = np.ndarray(shape=(size, size), dtype=int)
        known_maze.fill(UNSEEN)
        real_maze = np.ndarray(shape=(size, size), dtype=int)
        real_maze.fill(OPEN)

        update_maze(known_maze,real_maze,[15,15],[0,0])
        for i in range(size):
            if(i==15):
                continue
            val1 = known_maze[i][15]
            val2 = known_maze[15][i]
            self.assertEqual(val1,OPEN)
            self.assertEqual(val2,OPEN)