import operator
import types
from copy import copy
from functools import partial

import numpy
from deap import base, gp, tools, creator, algorithms

from consts import MAZE_SIZE, MAX_STEPS, UP, DOWN, LEFT, RIGHT
from gif_maker import make_gif
from mazes_creator.maze_consts import VISITED_POS, WALL, END, STRARTING_POSINGTION
from mazes_creator.maze_manager import update_maze, get_lsm_features, is_surrounded
from mazes_creator.maze_manager_ng import make_maze_manger_from_file

global maze
maze = None


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


def is_dead_end_left():
    return maze.is_dead_end_left(maze.current_pos)


def is_dead_end_right():
    return maze.is_dead_end_right(maze.current_pos)


def is_dead_end_up():
    return maze.is_dead_end_up(maze.current_pos)


def is_dead_end_down():
    return maze.is_dead_end_down(maze.current_pos)


def is_up_wall():
    return maze.is_up_wall()


def is_down_wall():
    return maze.is_down_wall()


def is_left_wall():
    return maze.is_left_wall()


def is_right_wall():
    return maze.is_right_wall()

def is_prev_pos_down():
    return maze.is_prev_pos_down()
def is_prev_pos_up():
    return maze.is_prev_pos_up()
def is_prev_pos_left():
    return maze.is_prev_pos_left()
def is_prev_pos_right():
    return maze.is_prev_pos_right()
def run_if_callable(input):
    res= input
    if isinstance(input,types.FunctionType):
        res = input()
    return res

def if_then_else(input, output1, output2):
    if input:
        return run_if_callable(output1)
    else:
        return run_if_callable(output2)

def get_valid_moves():
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
    if dir == UP:
        return is_dead_end_up()
    if dir == DOWN:
        return is_dead_end_down()
    if dir == LEFT:
        return is_dead_end_left()
    if dir==RIGHT:
        return is_dead_end_right()

def is_wall_dir(dir):
    if dir==UP:
        return is_up_wall()
    if dir==DOWN:
        return is_down_wall()
    if dir==LEFT:
        return is_left_wall()
    if dir==RIGHT:
        return is_right_wall()
def go_dir(dir):
    if dir==UP:
        return go_up()
    if dir==DOWN:
        return go_down()
    if dir==LEFT:
        return go_left()
    if dir==RIGHT:
        return go_right()
def get_dir_closest_to_exit():
    return maze.get_dir_closest_to_end()
def did_visit(dir):
    return maze.did_visit(dir)
pset = gp.PrimitiveSetTyped("MAIN", [], float, "IN")
# pset.addPrimitive(is_finished,[],bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)
# pset.addPrimitive(is_dead_end_down, [], bool)
# pset.addPrimitive(is_dead_end_up, [], bool)
# pset.addPrimitive(is_dead_end_right, [], bool)
# pset.addPrimitive(is_dead_end_left, [], bool)
pset.addPrimitive(get_dir_closest_to_exit,[],float)
pset.addPrimitive(is_dead_end_dir,[float],bool)
# pset.addPrimitive(get_valid_moves,[],list)
pset.addPrimitive(is_wall_dir,[float],bool)
pset.addPrimitive(did_visit,[float],bool)
# pset.addPrimitive(is_up_wall,[],bool)
# pset.addPrimitive(is_down_wall,[],bool)
# pset.addPrimitive(is_left_wall,[],bool)
# pset.addPrimitive(is_right_wall,[],bool)
pset.addPrimitive(is_wall_dir,[float],bool)
# pset.addPrimitive(is_prev_pos_down,[],bool)
# pset.addPrimitive(is_prev_pos_up,[],bool)
# pset.addPrimitive(is_prev_pos_left,[],bool)
# pset.addPrimitive(is_prev_pos_right,[],bool)

# pset.addPrimitive(prog3, [float, float, float], float)
# pset.addPrimitive(get_maze_manager,[MazeManager],MazeManager)
# pset.addPrimitive(go_dir,[float],bool)
# pset.addTerminal(go_up, float)
# pset.addTerminal(go_down, float)
# pset.addTerminal(go_left, float)
# pset.addTerminal(go_right, float)
# pset.addPrimitive(operator.lt, [float, float], bool)
# pset.addPrimitive(operator.eq, [float, float], bool)
# pset.addTerminal(MAZE_SIZE[0], float, 'size')
# pset.addTerminal(0, float, 'zero')
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
    score = 0
    for manger in managerers:
        valid_steps = 0
        visited_steps=0
        m = copy(manger)
        global maze
        maze = m

        for i in range(MAX_STEPS):
            try:
                try:
                    func = toolbox.compile(expr=individual)
                except Exception as e:
                    return -200,
                old_pos = copy(m.current_pos)
                res = func
                if maze.did_visit(res):
                    valid_steps -= 0.5
                s=go_dir(res)
                if s < 0:
                    score -= 100
                    break
                new_pos = m.current_pos
                if (old_pos[0] == new_pos[0] and old_pos[1] == new_pos[1]):
                    break
                else:
                    valid_steps+=1

                # if maze._get_maze_at_pos(maze.current_pos) !=
                score += s+valid_steps
                if m.is_finished():
                    score += 100,
            except Exception as e:
                score=-100
                break
        score -= m.dist_from_end(m.current_pos)
    if score>=200:
        pass
    return score,


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", eval_indv, managerers=[make_maze_manger_from_file(x) for x in range(2)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=30))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=30))


def main():
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.6, 0.01, 600, stats=mstats,
                                   halloffame=hof, verbose=True)
    return pop, log, hof

def make_gif_from_hof(hof):
    global manager
    m = make_maze_manger_from_file(1)
    manager = m
    mazes = []
    func = toolbox.compile(expr=hof.items[0])
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
    make_gif_from_hof(hof)
    print(hof)
