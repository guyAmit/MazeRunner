import operator

import deap
import numpy
from deap import base, gp, tools, creator, algorithms
from deap.gp import PrimitiveSet

from consts import MAZE_SIZE, MAX_STEPS
from mazes_creator.maze_consts import VISITED_POS, WALL, END, STRARTING_POSINGTION
from mazes_creator.maze_manager import update_maze, get_lsm_features, is_surrounded


def if_then_else(input, output1, output2):
    return output1 if input else output2

pset = PrimitiveSet("main", 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive()
pset.addTerminal(3)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def run_maze(model, maze, j):
    global SOLVED
    current_maze = maze[0]
    full_maze = maze[1]
    curr_pos = numpy.array(STRARTING_POSINGTION)
    prev_pos = curr_pos.copy()
    score = 0
    iligal_move = 0
    dead_end = 0
    features = numpy.zeros((1, MAX_STEPS, 6))
    mazes = []
    for i in range(MAX_STEPS):
        if model.net_type == 'cnn':
            dead_end = 1 if is_surrounded(
                current_maze, curr_pos) is not None else 0
            pred = model.predict(img=current_maze.reshape(
                (1, MAZE_SIZE[0], MAZE_SIZE[1], 1)),
                iligal_move=numpy.array([iligal_move]),
                dead_end=numpy.array([dead_end]))

        if model.net_type == 'lstm':
            features[:, i] = get_lsm_features(current_maze, curr_pos)
            if i != 0:
                features[:, i-1, pred] = -1
            pred = model.predict(features)
        iligal_move = 0
        if (curr_pos[0] + pred[0] >= current_maze.shape[0] or
            curr_pos[1] + pred[1] >= current_maze.shape[1] or
                curr_pos[0] + pred[0] < 0 or curr_pos[1] + pred[1] < 0):
            score += 5.5  # out of maze
            iligal_move = 1
            # return score
        elif current_maze[curr_pos[0] + pred[0], curr_pos[1]+pred[1]] == END:
            score -= 1000  # maze ending bonus
            print('finished maze !!')
            if j not in SOLVED:
                # make_gif(mazes, j)
                SOLVED.add(j)
                return score+i*3
            return score+i*10
        elif current_maze[curr_pos[0] + pred[0], curr_pos[1]+pred[1]] == WALL:
            score += 5  # run into wall
            iligal_move = 1
            # return score
        elif current_maze[curr_pos[0] + pred[0],
                          curr_pos[1]+pred[1]] == VISITED_POS:
            score -= 2
            prev_pos = curr_pos.copy()
            curr_pos[0] += pred[0]
            curr_pos[1] += pred[1]
        else:
            score -= 5
            prev_pos = curr_pos.copy()
            curr_pos[0] += pred[0]
            curr_pos[1] += pred[1]

        new_tiels = update_maze(current_maze, full_maze, new_pos=curr_pos,
                                old_pos=prev_pos)
        # converted_maze = convert_array(current_maze)
        # mazes.append(converted_maze)
        if new_tiels > 0:
            score -= 15

    del maze
    return score


def reward_func(mazes, model):
    def get_reward(weights):
        model.set_weights(weights)
        reward = 0
        counter = 0
        for maze in mazes:
            reward += run_maze(model, [maze[0].copy(), maze[1]], counter)
            counter += 1
        print(reward)
        return -(reward)
    return get_reward

def eval_indv(individual, mazes):
    pass
toolbox.register("evaluate", eval_indv, points=[x/10. for x in range(-10,10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                               halloffame=hof, verbose=True)