import operator
import pickle
import timeit
import types
from copy import copy
from functools import partial
from multiprocessing.spawn import freeze_support

import numpy
from deap import base, gp, tools, creator, algorithms
import multiprocessing
from consts import MAZE_SIZE, MAX_STEPS, UP, DOWN, LEFT, RIGHT
from gif_maker import make_gif
from mazes_creator.maze_consts import VISITED_POS, WALL, END, STRARTING_POSINGTION
from mazes_creator.maze_manager import update_maze, get_lsm_features, is_surrounded
from mazes_creator.maze_manager_ng import make_maze_manger_from_file, MazeManager
import matplotlib.pyplot as plt
import numpy as np

global maze
maze = None
global solved_mazes
solved_mazes=set()
global n_gen
n_gen=0
global indv_count
indv_count =0
# def if_then_else(input, output1, output2):
#     return output1 if input else output2


def go_up():
    global maze
    return maze.go_up()


def go_left():
    global maze
    return maze.go_left()


def go_right():
    global maze
    return maze.go_right()


def go_down():
    global maze
    return maze.go_down()


def is_finished():
    global maze
    return maze.is_finished()

def get_current_direction():
    global maze
    return maze.get_current_direction()
def is_dead_end_left():
    global maze
    return maze.is_dead_end_left(maze.current_pos)


def is_dead_end_right():
    global maze
    return maze.is_dead_end_right(maze.current_pos)


def is_dead_end_up():
    global maze
    return maze.is_dead_end_up(maze.current_pos)


def is_dead_end_down():
    global maze
    return maze.is_dead_end_down(maze.current_pos)


def is_up_wall():
    global maze
    return maze.is_up_wall()


def is_down_wall():
    global maze
    return maze.is_down_wall()


def is_left_wall():
    global maze
    return maze.is_left_wall()


def is_right_wall():
    global maze
    return maze.is_right_wall()

def is_prev_pos_down():
    global maze
    return maze.is_prev_pos_down()
def is_prev_pos_up():
    global maze
    return maze.is_prev_pos_up()
def is_prev_pos_left():
    global maze
    return maze.is_prev_pos_left()
def is_prev_pos_right():
    global maze
    return maze.is_prev_pos_right()
def run_if_callable(input):
    res= input
    if isinstance(input,types.FunctionType):
        res = input()
    return res
def get_most_valuable_dir():
    global maze
    return maze.get_most_valuable_dir()
def if_then_else(input, output1, output2):
    if input:
        return run_if_callable(output1)
    else:
        return run_if_callable(output2)
def get_least_visited_pos():
    global maze
    return maze.get_least_visited_pos()
def get_valid_moves():
    global maze
    return maze.get_valid_moves()

def progn(*args):
    for arg in args:
        arg()

def prog3(c1,c2,c3):
    c1()
    c2()
    return c3()
# def prog3(out1, out2, out3):
#     return partial(progn, out1, out2, out3)

def is_dead_end_dir(dir):
    global maze
    if dir == UP:
        return is_dead_end_up()
    if dir == DOWN:
        return is_dead_end_down()
    if dir == LEFT:
        return is_dead_end_left()
    if dir==RIGHT:
        return is_dead_end_right()

def is_wall_dir(dir):
    global maze
    if dir==UP:
        return is_up_wall()
    if dir==DOWN:
        return is_down_wall()
    if dir==LEFT:
        return is_left_wall()
    if dir==RIGHT:
        return is_right_wall()
def go_dir(dir):
    global maze
    if dir==UP:
        return go_up()
    if dir==DOWN:
        return go_down()
    if dir==LEFT:
        return go_left()
    if dir==RIGHT:
        return go_right()
def get_dir_closest_to_exit():
    global maze
    return maze.get_dir_closest_to_end()
def did_visit(dir):
    global maze
    return maze.did_visit(dir)

pset = gp.PrimitiveSetTyped("MAIN", [], float, "IN")
pset.addPrimitive(if_then_else, [bool, float, float], float)

pset.addPrimitive(is_dead_end_dir,[float],bool)
pset.addPrimitive(is_wall_dir,[float],bool)
pset.addPrimitive(did_visit,[float],bool)

pset.addTerminal(get_least_visited_pos,float)
pset.addTerminal(get_current_direction,float)
pset.addTerminal(get_most_valuable_dir,float)
pset.addTerminal(get_dir_closest_to_exit,float)

# pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)

pset.addTerminal(UP,float,'up')
pset.addTerminal(DOWN,float,'down')
pset.addTerminal(LEFT,float,'left')
pset.addTerminal(RIGHT,float,'right')

pset.addTerminal(True, bool, 'true')
pset.addTerminal(False, bool, 'false')



# def reward_func(mazes, model):
#     def get_reward(weights):
#         model.set_weights(weights)
#         reward = 0
#         counter = 0
#         for maze in mazes:
#             reward += run_maze(model, [maze[0].copy(), maze[1]], counter)
#             counter += 1
#         print(reward)
#         return -(reward)
#
#     return get_reward


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def eval_indv(individual, managerers):
    global maze, solved_mazes, n_gen, indv_count
    indv_count+=1
    if indv_count%100==0:
        n_gen+=1
    score = 0
    counter = 0
    current_solved_mazes =set()
    for manger in managerers:
        # steps_count= min(8*n_gen, MAX_STEPS)
        steps_count=MAX_STEPS
        valid_steps = 0
        visited_steps=0
        maze = MazeManager(manger.full_maze,manger.known_maze)
        counter+=1
        for i in range(steps_count):
            try:
                try:
                    if type(individual) is int:
                        res= individual
                    else:
                        func = toolbox.compile(expr=individual)
                        res = func
                        if type(res) is not int:
                            res = func()
                except Exception as e:
                    return -999999,
                old_pos = copy(maze.current_pos)

                if maze.did_visit(res):
                    nov= maze.number_of_visits(res)
                    visited_steps +=nov*2
                    if nov>5:
                        score-=500
                        break
                s=go_dir(res)

                if s < 0:
                    score -= 600
                    break
                new_pos = maze.current_pos
                if (old_pos[0] == new_pos[0] and old_pos[1] == new_pos[1]):
                    score-=600
                    break
                else:
                    valid_steps+=1

                score += s+valid_steps - visited_steps
                if maze.is_close_to_finish():
                    current_solved_mazes.add(counter)
                    score += 8*(MAX_STEPS-i)
                    if counter not in solved_mazes:
                        solved_mazes.add(counter)
                        # plt.imshow(-np.array(list(maze.known_maze)))
                        # plt.show()
                        # score+=100
                        print(f". maze {counter}!")
                    break
            except Exception as e:
                score=-600
                break
        score -= 2*maze.dist_from_end(maze.current_pos)
    score=score -individual.height
    # if len(solved_mazes)>0:
    solved = len(current_solved_mazes)
    total = len(managerers)
    unsolved = total-solved
    if unsolved==0:
        coef=total
    else:
        coef = total/unsolved
    score=score*coef
    return score,


toolbox = base.Toolbox()

toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))


def evaluate_on_test(individual, test_mazes):
    global maze
    counter =0
    solved = set()
    for m in test_mazes:
        counter+=1
        for i in range(MAX_STEPS):
                try:
                    func = toolbox.compile(expr=individual)
                    if maze.is_close_to_finish():
                        solved.add(counter)
                        break
                except:
                    pass
    print(f"managed to solve {len(solved)} out of {len(test_mazes)}")
def get_fit(ind):
    res = ind.fitness.values[0]
    return res
def main():
    # freeze_support()
    # pool = multiprocessing.Pool()
    # toolbox.register("map", pool.map)
    global n_gen, solved_mazes
    hof = tools.HallOfFame(1)



    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 8))
    for i in range(3):
        n_gen=0
        solved_mazes=set()
        mstats = tools.Statistics(get_fit)
        stats_size = tools.Statistics(len)
        # mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("Avg", numpy.mean)
        mstats.register("Std", numpy.std)
        mstats.register("Min", numpy.min)
        mstats.register("Max", numpy.max)

        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("evaluate", eval_indv, managerers=[make_maze_manger_from_file(x) for x in range(20)])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        start = timeit.default_timer()
        pop = toolbox.population(n=500)
        ngen=200
        pop, logbook = algorithms.eaSimple(pop, toolbox, 0.7, 0.1, 100, stats=mstats,
                                       halloffame=hof, verbose=True)
        # pickle.dump(hof[0], f'pickle_hof_{i}', -1)
        evaluate_on_test(hof.items[0], [make_maze_manger_from_file(x) for x in range(20,25)])
        end = timeit.default_timer()
        interval = end - start
        print(f'Time for termination: {interval} sec')
        print(f'Avg time for generation: {interval / ngen} sec')

        gen = logbook.select("gen")
        fit_mins = logbook.select("Min")
        fit_avgs = logbook.select("Avg")
        fit_maxs = logbook.select("Max")
        fit_std = logbook.select("Std")

        ax[i].plot(gen, fit_mins, "b-", label="Minimum Fitness")
        ax[i].plot(gen, fit_avgs, "r-", label="Average Fitness")
        ax[i].plot(gen, fit_maxs, "g-", label="Max Fitness")
        ax[i].plot(gen, fit_std, "y-", label="std Fitness")
        ax[i].set_xlabel("Generation", fontsize=18)
        ax[i].tick_params(labelsize=16)
        ax[i].set_ylabel("Fitness", color="b", fontsize=18)
        ax[i].legend(loc="lower right", fontsize=14)
        ax[i].set_title(f'Avg time for generation {interval:.4f} sec\n Time for termination {interval / ngen:.4f} sec',
                        fontsize=18)
    plt.show()
    return pop, logbook, hof

def make_gif_from_hof(hof):
    global manager

    m = make_maze_manger_from_file(1)
    manager = m
    mazes = []
    func = toolbox.compile(expr=hof)
    mazes.append(m.known_maze)
    for i in range(MAX_STEPS):
        try:
            res = func()
        except:
            pass
        mazes.append(m.known_maze)
        if manager.is_finished():
            break
    make_gif(mazes,1)

if __name__ == "__main__":
    pop, log, hof =main()
    make_gif_from_hof(hof.items[0])
    pickle.dump(hof[0], 'pickle_hof', -1)
    print(hof)
