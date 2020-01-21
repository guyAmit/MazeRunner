import datetime
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np

def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(output_file, images, duration=duration)


def make_gif(mazes):
    output_dir = 'Gif-%s' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    os.mkdir(output_dir)
    names = []

    for i in range(len(mazes)):
        imname = f'{output_dir}\\{i}.jpg'
        m = np.kron(mazes[i], np.ones((100,100), dtype=np.int8))
        plt.imsave(imname, m)
        names.append(imname)
    create_gif(names, 0.5)

if __name__ == '__main__':
    names = []
    for i in range(20):
        imname = f'C:\\Users\\liorbass\\Desktop\\EvoProject\\maze_runner\\Gif-2020-18-21-17-18-46\\{i}.jpg'
        names.append(imname)
    create_gif(names, 0.5)