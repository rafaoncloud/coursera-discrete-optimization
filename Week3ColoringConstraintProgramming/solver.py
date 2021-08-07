#!/usr/bin/python
# -*- coding: utf-8 -*-
import collections
import sys
import random
import math

random.seed(123)


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    # Initialize 2d list of empty columns
    edges = [[] for i in range(node_count)]

    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        a = int(parts[0])
        b = int(parts[1])
        edges[a].append(b)
        edges[b].append(a)

    # build a trivial solution
    # every node has its own color
    # solution, obj_value, is_optimal = trivial_solution(node_count, edge_count, edges)
    solution, obj_value, is_optimal = search(node_count, edge_count, edges)

    # prepare the solution in the specified output format
    output_data = str(obj_value) + ' ' + str(is_optimal) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def trivial_solution(node_count, edge_count, edges):
    # build a trivial solution
    # every node has its own color
    solution = range(0, node_count)
    obj_value = node_count

    return solution, obj_value, 0


# ++++++++++++++++++++++++++++
# TABU SEARCH
# ++++++++++++++++++++++++++++


class Tabu:

    def __init__(self, tabu_size):
        self.tabu_hash = set()
        self.tabu_queue = collections.deque()
        self.tabu_size = tabu_size

    def contains(self, node):
        return node in self.tabu_hash

    def push(self, node):
        if self.contains(node): return
        self.tabu_hash.add(node)
        self.tabu_queue.appendleft(node)
        if len(self.tabu_hash) > self.tabu_size: self.pop()

    def pop(self):
        top = self.tabu_queue.pop()
        self.tabu_hash.remove(top)


def select_next_node(violation, colors, tabu):
    max_violation = - sys.maxsize
    max_violation_node = []
    for cur_node in range(len(violation)):
        if violation[cur_node] == 0: continue
        if tabu.contains(cur_node): continue
        if max_violation == violation[cur_node]: max_violation_node.append(cur_node)
        elif max_violation < violation[cur_node]:
            max_violation = violation[cur_node]
            max_violation_node.clear()
            max_violation_node.append(cur_node)
    if len(max_violation_node) == 0: return -1
    return random.choice(max_violation_node)


def change_color(node, node_neighbors, colors, total_colors_count, violation, total_violation):
    color_count = [0] * total_colors_count
    for neighbor in node_neighbors:
        neighbor_color = colors[neighbor]
        color_count[neighbor_color] += 1
    min_color_count = sys.maxsize
    min_color = []
    for cur_color in range(0, total_colors_count):
        if cur_color == colors[node]: continue
        if min_color_count > color_count[cur_color]:
            min_color_count = color_count[cur_color]
            min_color.clear()
            min_color.append(cur_color)
        elif min_color_count == color_count[cur_color]:
            min_color.append(cur_color)
    assert len(min_color) > 0
    new_color = random.choice(min_color)
    for neighbor in node_neighbors:
        if colors[neighbor] == colors[node]:
            violation[neighbor] -= 1
            violation[node] -= 1
            total_violation -= 2
        elif colors[neighbor] == new_color:
            violation[neighbor] += 1
            violation[node] += 1
            total_violation += 2
    colors[node] = new_color
    return total_violation

def remove_color(colors, total_colors_count):
    to_remove = random.randint(0, total_colors_count - 1)
    new_colors = []
    for c in colors:
        if c == to_remove: new_colors.append(random.randint(0, total_colors_count - 2))
        elif c > to_remove: new_colors.append(c - 1)  # Remove color c - 1
        else: new_colors.append(c);
    return new_colors


def init_violation(edges, colors):
    violation = []
    total_violation = 0
    for cur_node in range(0, len(edges)):
        cur_violation = 0
        for neighbor in edges[cur_node]:
            cur_violation += colors[cur_node] == colors[neighbor]
        violation.append(cur_violation)
        total_violation += cur_violation
    return violation, total_violation


def _tabu_search(edges, colors, total_colors_count, tabu_size):
    step_limit = 50000
    step_count = 0

    violation, total_violation = init_violation(edges, colors)

    # Tabu hash table and tabu queue, they contain same data
    # Queue FIFO (First In First Out)
    tabu = Tabu(tabu_size)

    while step_count < step_limit and total_violation > 0:
        node = select_next_node(violation, colors, tabu)
        while node == -1:
            tabu.pop()
            node = select_next_node(violation, colors, tabu)
        tabu.push(node)
        total_violation = change_color(node, edges[node], colors, total_colors_count, violation, total_violation)
        step_count += 1
    return total_violation == 0, step_count


def search(node_count, edge_count, edges):
    colors = [0] * node_count
    total_colors_count = len(colors)

    colors = initial_solution(colors, total_colors_count)

    tabu_size_limit = math.ceil(node_count / 5)
    retry_limit = 100

    feasible_color = []
    feasible_color_count = -1

    for cur_colors_count in range(total_colors_count, 1, -1):
        retry_count = 0
        while True:
            feasible, step_count = _tabu_search(edges, colors, cur_colors_count, tabu_size_limit)
            if feasible:
                feasible_color = colors
                feasible_color_count = cur_colors_count
                colors = remove_color(feasible_color, feasible_color_count)
                break

            retry_count += 1
            if retry_count <= retry_limit:
                return feasible_color, feasible_color_count, 0
            colors = remove_color(feasible_color, feasible_color_count)

    solution = feasible_color
    obj_value = feasible_color_count
    is_optimal = 0

    return solution, obj_value, is_optimal


def initial_solution(colors, total_colors_count):
    new_colors = [0] * len(colors)
    for i in range(len(new_colors)):
        new_colors[i] = random.randint(0, total_colors_count - 1)
    return new_colors


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
