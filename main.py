from daedalus import Maze
import matplotlib.pyplot as plt
import numpy as np

def make_maze(size, func):
    maze = Maze(*size)
    # func(maze, nEntrancePos=0, nRndBias=2)
    m =Maze.create_wilson(maze, nEntrancePos=0, nRndBias=2)
    return maze

def show_maze(maze):
    print('Maze entrance: {}'.format(maze.entrance))
    print('Maze exit: {}'.format(maze.exit))
    a = np.array(list(maze))
    plt.imshow(-a)
    plt.show()
    pass
def main():
    SIZE = 31
    m = make_maze((SIZE,SIZE), Maze.create_kruskal)
    show_maze(m)

if __name__=='__main__':
    main()