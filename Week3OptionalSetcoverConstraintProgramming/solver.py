#!/usr/bin/python
# -*- coding: utf-8 -*-

# The MIT License (MIT)
#
# Copyright (c) 2014 Carleton Coffrin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from collections import namedtuple

from ortools.sat.python import cp_model

Set = namedtuple("Set", ['index', 'cost', 'items'])


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    item_count = int(parts[0])
    set_count = int(parts[1])
    
    sets = []
    for i in range(1, set_count+1):
        parts = lines[i].split()
        sets.append(Set(i-1, int(parts[0]), list(map(int, parts[1:]))))

    #solution, obj_value, is_optimal = trivial_solution(item_count, set_count, sets)
    solution, obj_value, is_optimal = ortools_cp_sat(item_count, set_count, sets)

    # prepare the solution in the specified output format
    output_data = str(obj_value) + ' ' + str(is_optimal) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def trivial_solution(item_count, set_count, sets):

    # build a trivial solution
    # pick add sets one-by-one until all the items are covered
    solution = [0]*set_count
    coverted = set()
    
    for s in sets:
        solution[s.index] = 1
        coverted |= set(s.items)
        if len(coverted) >= item_count:
            break
        
    # calculate the cost of the solution
    obj_value = sum([s.cost*solution[s.index] for s in sets])

    return solution, obj_value, 0


def ortools_cp_sat(item_count, set_count, sets):

    # [START model]
    model = cp_model.CpModel()
    # [END model]

    # [START variables]
    variables = []
    for i in range(set_count):
        x = model.NewIntVar(0, 1, 'x_%d' % i)
        variables.append(x)
    # [END variables]

    # [START constraints]
    set_items = []  # Row (sets/fire stations) Columns (items/regions)
    for set_index in range(set_count):
        set_items.append([0] * item_count) 
        for item_index in range(item_count):
            if item_index in sets[set_index].items:
                set_items[set_index][item_index] = 1
    set_items_transposed = transpose(set_items)  # Row (items/regions) Col (sets/fire stations)
    for items in set_items_transposed:
        # Each item/region must be covered at least once
        model.Add(cp_model._ScalProd(variables, items) >= 1)
    # [END constraints]

    # [START objective function]
    costs = []
    for s in sets:
        costs.append(s.cost)
    # Each
    model.Minimize(cp_model._ScalProd(variables, costs))
    # [END objective function]

    # [START solve]
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model)
    # [END solve]

    solution = set_count * [0]
    obj_value = solver.ObjectiveValue()
    is_optimal = 0

    # [START solution]
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        is_optimal = 1 if status == cp_model.OPTIMAL else 0
        for i in range(set_count):
            solution[i] = solver.Value(variables[i])
    elif status == cp_model.INFEASIBLE:
        print('INFEASIBLE')
    # [END solution]

    return solution, obj_value, is_optimal


def transpose(list_2d):
    return [[row[i] for row in list_2d] for i in range(len(list_2d[0]))]


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/sc_6_1)')

