import datetime
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np


def create_gif(filenames, duration, maze_number):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = (f'gif-Maze-{str(maze_number)}-%s.gif' %
                   datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S'))
    imageio.mimsave(output_file, images, duration=duration)


def make_gif(mazes, maze_number):
    output_dir = (f'Maze_number-{str(maze_number)}-%s' %
                  datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S'))
    os.mkdir(output_dir)
    names = []

    for i in range(len(mazes)):
        imname = os.path.join(output_dir, f'{i}.jpg')
        m = np.kron(mazes[i], np.ones((100, 100), dtype=np.int8))
        plt.imsave(imname, m)
        names.append(imname)
    create_gif(names, 0.5, maze_number)
