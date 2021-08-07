#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import math
import random
import time
from collections import namedtuple

random.seed(123)

Location = namedtuple("Location", ['x', 'y'])


class Edge:
    def __init__(self):
        self.from_loc = None
        self.to_loc = None

    def __init__(self, from_loc = None, to_loc = None):
        self.from_loc = from_loc
        self.to_loc = to_loc

    def copy(self):
        return Edge(self.from_loc, self.to_loc)

    @staticmethod
    def copy_list(edges):
        return [edge.copy() for edge in edges]


class CostMatrix:
    def __init__(self, locations):
        self.cost_matrix = self._build_cost_matrix(locations)

    @staticmethod
    def _build_cost_matrix(locations):
        cost_matrix = []
        for loc_from in locations:
            vector = []
            for loc_to in locations:
                vector.append(CostMatrix._euclidean_distance(loc_from, loc_to))
            cost_matrix.append(vector)

        return cost_matrix

    @staticmethod
    def _euclidean_distance(a, b):
        return length(a, b)

    def dist(self, point_from, point_to):
        return self.cost_matrix[point_from][point_to]


class Penalty:
    def __init__(self, num_locations):
        self.penalty = [[0] * num_locations for _ in range(num_locations)]

    def get(self, i, j):
        if i < j: i, j = j, i
        return self.penalty[i][j]

    def increment(self, i, j):
        if i < j: i, j = j, i
        self.penalty[i][j] += 1


class Activate:
    def __init__(self, size):
        self.bits = [1] * size
        self.ones = size

    def set_1(self, i):
        self.ones += 1 if self.bits[i] == 0 else 0
        self.bits[i] = 1

    def set_0(self, i):
        self.ones -= 1 if self.bits[i] == 1 else 0
        self.bits[i] = 0

    def get(self, i):
        return self.bits[i]

    def size(self):
        return len(self.bits)


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def swap(vector, i, j):
    vector[j], vector[i] = vector[i], vector[j]


def solution_cost(cm, edges):
    # Calculates the cost of the route
    cost = .0
    for i in range(0, len(edges)):
        cost += cm.dist(i, edges[i].to_loc)
    return cost


def str_route(edges):
    output = ""
    location = 0
    for i in range(0, len(edges)):
        output += str(location)
        output += '\n' if i + 1 == len(edges) else ' '
        location = edges[location].to_loc
    return output


def construction_heuristic(cm, num_locations):
    route = []
    for i in range(0, num_locations):
        route.append(i)
    for i in range(0, len(route) - 1):
        min_distance = sys.maxsize
        min_distance_location = -1
        for j in range(i + 1, len(route)):
            distance = cm.dist(route[i], route[j])
            if distance < min_distance:
                min_distance = distance
                min_distance_location = j
        swap(route, i + 1, min_distance_location)
    edges = [Edge() for _ in range(len(route))]
    for i in range(0, len(route)):
        location = route[i]
        next_location = route[(i + 1) % num_locations]
        edges[location].to_loc = next_location
        edges[next_location].from_loc = location
    return edges


def total_augmented_cost(cm, edges, penalty, gls_lambda):
    augmented_cost = 0.0
    for i in range(0, len(edges)):
        i_to_loc = edges[i].to_loc
        c = cm.dist(i, i_to_loc)
        p = penalty.get(i, i_to_loc)
        augmented_cost += c + p * gls_lambda
    return augmented_cost


def init_lambda(cm, edges, gls_alpha):
    return gls_alpha * solution_cost(cm, edges) / len(edges)


# Swap 2 edges, with 4 vertexes (locations): t1, t2, t3, t4
# Before swapping 2 edges: t1 -> t2, t3 -> t4
# After swapping 2 edges: t1 -> t3, t2 -> t4

def select_t3_t4(t1, t2, cm, edges, penalty, gls_lambda):
    max_gain = - sys.maxsize
    t4_candidate = []
    t2_to_loc = edges[t2].to_loc
    for i in range(len(edges)):
        t4 = i
        t3 = edges[t4].from_loc
        if t4 == t1 or t4 == t2 or t4 == t2_to_loc: continue
        c12 = cm.dist(t1, t2)
        c34 = cm.dist(t3, t4)
        c13 = cm.dist(t1, t3)
        c24 = cm.dist(t2, t4)
        p12 = penalty.get(t1, t2)
        p34 = penalty.get(t3, t4)
        p13 = penalty.get(t1, t3)
        p24 = penalty.get(t2, t4)
        gain = (c12 + gls_lambda + p12) + (c34 + gls_lambda + p34) - (c13 + gls_lambda + p13) - (c24 + gls_lambda + p24)
        if gain > max_gain:
            max_gain = gain
            t4_candidate.clear()
            t4_candidate.append(t4)
        elif max_gain == gain:
            t4_candidate.append(t4)
    if max_gain > 0:
        t4 = random.choice(t4_candidate)
        t3 = edges[t4].from_loc
        return t3, t4
    return -1, -1


def swap_edge(t1, t2, t3, t4, cm, edges, penalty, cost, augmented_cost, gls_lambda):
    cur_location = t2
    cur_location_to_loc = edges[cur_location].to_loc
    while cur_location != t3:
        next_cur_loc = cur_location_to_loc
        next_cur_loc_to_loc = edges[cur_location_to_loc].to_loc
        edges[cur_location].from_loc = cur_location_to_loc
        edges[cur_location_to_loc].to_loc = cur_location
        cur_location = next_cur_loc
        cur_location_to_loc = next_cur_loc_to_loc
    edges[t2].to_loc = t4
    edges[t4].from_loc = t2
    edges[t1].to_loc = t3
    edges[t3].from_loc = t1
    c12 = cm.dist(t1, t2)
    c34 = cm.dist(t3, t4)
    c13 = cm.dist(t1, t3)
    c24 = cm.dist(t2, t4)
    p12 = penalty.get(t1, t2)
    p34 = penalty.get(t3, t4)
    p13 = penalty.get(t1, t3)
    p24 = penalty.get(t2, t4)
    gain = (c12 + gls_lambda + p12) + (c34 + gls_lambda + p34) - (c13 + gls_lambda + p13) - (c24 + gls_lambda + p24)
    cost -= c12 + c34 - c13 - c24
    augmented_cost -= gain
    return cost, augmented_cost


def add_penalty(cm, edges, penalty, active, augmented_cost, gls_lambda):
    max_util = - sys.maxsize
    max_util_location = []
    for i in range(len(edges)):
        i_to_loc = edges[i].to_loc
        c = cm.dist(i, i_to_loc)
        p = 1 + penalty.get(i, i_to_loc)
        util = c / (1 + p)
        if max_util < util:
            max_util = util
            max_util_location.clear()
            max_util_location.append(i)
        elif max_util == util: max_util_location.append(i)
    for i in max_util_location:
        i_to_loc = edges[i].to_loc
        penalty.increment(i, i_to_loc)
        active.set_1(i)
        active.set_1(i_to_loc)
        augmented_cost += gls_lambda
    return augmented_cost


def guided_local_search(cm, edges):
    """
    :param cm: CostMatrix
    :param edges: Initial solution
    :return: Best solution found
    """
    step_limit = 100000 if len(edges) <= 1000 else 100
    penalty = Penalty(len(edges))
    gls_alpha = 0.1
    gls_lambda = 0.0
    active = Activate(len(edges))
    cur_edges = edges
    cur_cost = solution_cost(cm, edges)
    cur_augmented_cost = total_augmented_cost(cm, cur_edges, penalty, gls_lambda)
    best_edges = cur_edges
    best_cost = cur_cost

    MAX_STALLED_ITERATIONS = 64
    start_time = time.perf_counter()

    for cur_step in range(0, step_limit):
        time_elapsed = time.perf_counter() - start_time
        print('[Step=%d/%d][Timer=%f][CurCost=%f][CurAugmentedCost=%f][BestCost=%f][Lambda=%f][Alpha=%f]' % (cur_step + 1, step_limit, time_elapsed,  cur_cost, cur_augmented_cost, best_cost, gls_lambda, gls_alpha))
        stalled = 0
        while active.ones > 0:
            stalled += 1
            if stalled >= MAX_STALLED_ITERATIONS:
                print('[FORCE EXIT STEP] Max stalled iterations of 1024 achieved in the current step')
                break
            elif time_elapsed > 600:
                return best_edges
            for bit in range(0, active.size()):
                if active.ones == 0: break
                if not active.get(bit): continue
                bit_from_loc = cur_edges[bit].from_loc
                bit_to_loc = cur_edges[bit].to_loc
                t1_t2_candidate = []
                t1_t2_candidate.append([bit_from_loc, bit])
                t1_t2_candidate.append([bit, bit_to_loc])
                for j in range(len(t1_t2_candidate)):
                    t1, t2 = t1_t2_candidate[j]
                    t3, t4 = select_t3_t4(t1, t2, cm, cur_edges, penalty, gls_lambda)
                    if t3 == -1:
                        if j == 1: active.set_0(bit)
                        continue
                    cur_cost, cur_augmented_cost = swap_edge(t1, t2, t3, t4, cm, cur_edges, penalty, cur_cost, cur_augmented_cost, gls_lambda)
                    cur_cost = solution_cost(cm, cur_edges)
                    active.set_1(t1)
                    active.set_1(t2)
                    active.set_1(t3)
                    active.set_1(t4)
                    break
                if cur_cost < best_cost:
                    best_edges = Edge.copy_list(edges)
                    best_cost = cur_cost
        if gls_lambda == 0.0: gls_lambda = init_lambda(cm, edges, gls_alpha)
        cur_augmented_cost = add_penalty(cm, cur_edges, penalty, active, cur_augmented_cost, gls_lambda)
    return best_edges


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    num_locations = int(lines[0])

    locations = []
    for i in range(1, num_locations + 1):
        line = lines[i]
        parts = line.split()
        locations.append(Location(float(parts[0]), float(parts[1])))

    print('Number of locations: %d' % len(locations))

    if num_locations > 2000:
        return "None"

    print('Creating cost matrix with euclidean distance...')
    cm = CostMatrix(locations)

    print('Creating initial solution...')
    # solution, obj_value, is_optimal = trivial_algorithm(locations, num_nodes)
    edges = construction_heuristic(cm, num_locations)  # Nearest Neighbors

    print('Start solving with the local search algorithm: Guided Local Search (GLS)')
    edges = guided_local_search(cm, edges)

    solution = str_route(edges)
    cost = solution_cost(cm, edges)
    is_optimal = 0

    # prepare the solution in the specified output format
    output_data = '%.2f' % cost + ' ' + str(is_optimal) + '\n'
    output_data += solution

    return output_data


def trivial_algorithm(locations, num_locations):
    # build a trivial solution
    # visit the nodes in the order they appear in the file
    edges = range(0, num_locations)

    # calculate the length of the tour
    obj = length(locations[edges[-1]], locations[edges[0]])
    for index in range(0, num_locations - 1):
        obj += length(locations[edges[index]], locations[edges[index + 1]])

    return edges


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')
