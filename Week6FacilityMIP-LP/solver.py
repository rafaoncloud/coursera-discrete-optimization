#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from collections import namedtuple
import math
import sys
import random
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model  # CP-SAT solver

random.seed(123)

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

FACTOR = 100000000


class CostMatrix:
    def __init__(self, facilities, customers):
        self.cost_matrix = []
        for f in facilities:
            vector = [length(f.location, c.location) for c in customers]
            self.cost_matrix.append(vector)

    def get(self, f, c):
        return self.cost_matrix[f][c]

    def get_cost_matrix_as_int(self):
        cost_matrix_int = []
        for vector in self.cost_matrix:
            vector_int = [int(dist * FACTOR) for dist in vector]
            cost_matrix_int.append(vector_int)
        return cost_matrix_int

    def get_cost_matrix_int_as_vector(self):
        cost_matrix_int = []
        for vector in self.cost_matrix:
            vector_int = [int(dist * FACTOR) for dist in vector]
            cost_matrix_int.extend(vector_int)
        return cost_matrix_int


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def transpose(list_2d):
    return [[row[i] for row in list_2d] for i in range(len(list_2d[0]))]


def solve_ortools_mip(cm: CostMatrix, facilities, customers, time_limit_seconds, num_threads=None):
    # [START data]
    print('Facilities=%d Customers=%d' % (len(facilities), len(customers)))
    # [END data]

    solver = pywraplp.Solver('SolveIntegerProblem', pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)

    # [START variables]
    x = []
    y = [[] for x in range(len(facilities))]
    for f in range(len(facilities)):
        x.append(solver.BoolVar('x_%d' % f))
        for c in range(len(customers)):
            y[f].append(solver.BoolVar('y_%d_%d' % (f, c)))
    # [END variables]

    # [START constraints]
    # A customer has exactly 1 assigned facility
    for c in range(len(customers)):
        constraint: pywraplp.Constraint = solver.Constraint(1.0, 1.0)
        for f in range(len(facilities)):
            constraint.SetCoefficient(y[f][c], 1.0)
    # The demand of the assigned customer cannot exceed the maximum capacity of the facility
    for f in range(len(facilities)):
        constraint: pywraplp.Constraint = solver.Constraint(0.0, facilities[f].capacity)
        for c in range(len(customers)):
            constraint.SetCoefficient(y[f][c], customers[c].demand)
    # Customer can only be assigned to facility if the facility x(f) is assigned
    for f in range(len(facilities)):
        for c in range(len(customers)):
            constraint: pywraplp.Constraint = solver.Constraint(-solver.infinity(), 0.0)
            constraint.SetCoefficient(y[f][c], 1.0)
            constraint.SetCoefficient(x[f], -1.0)
    # [END constraints]

    # [START objective function]
    objective = solver.Objective()
    objective.SetMinimization()
    for f in range(len(facilities)):
        objective.SetCoefficient(x[f], facilities[f].setup_cost)
        for c in range(len(customers)):
            objective.SetCoefficient(y[f][c], cm.get(f, c))
    # [END objective function]

    # [START solve]
    print('Variables=%d Constraints=%d' % (solver.NumVariables(), solver.NumVariables()))
    solver.SetTimeLimit(1000 * time_limit_seconds)
    if num_threads is not None: solver.SetNumThreads(num_threads)
    solver.EnableOutput()
    status = solver.Solve()
    print('Status=%d Iterations=%d Wall-Time=%d' % (status, solver.wall_time(), solver.iterations()))
    # [END solve]

    # [START solution]
    solution = [-1] * len(customers)
    cost = sys.maxsize
    is_optimal = 0
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        cost = solver.Objective().Value()
        is_optimal = 1 if status == pywraplp.Solver.OPTIMAL else 0
        y_assigned = [[] for x in range(len(facilities))]
        for f in range(len(facilities)):
            for c in range(len(customers)):
                y_assigned[f].append(int(y[f][c].solution_value()))
                if y_assigned[f][c] == 1: solution[c] = f
        #print(y_assigned)
        print(solution)
    else:
        print('INFEASIBLE')
    # [END solution]

    return solution, cost, is_optimal


def solve_ortools_cp_sat(cm: CostMatrix, facilities, customers):
    # [START data]
    demands = [c.demand for c in customers]
    capacities = [f.capacity for f in facilities]
    fixed_costs = [int(f.setup_cost) for f in facilities]
    fixed_costs_factor = [int(f.setup_cost * FACTOR) for f in facilities]
    dist = cm.get_cost_matrix_as_int()
    dist_as_vector = cm.get_cost_matrix_int_as_vector()
    print('Facilities=%d Customers=%d' % (len(facilities), len(customers)))
    print('CUSTOMER')
    print('demands=%s' % demands)
    print('FACILITIES')
    print('capacities=%s' % capacities)
    print('fixed_costs=%s' % fixed_costs)
    print('COST MATRIX')
    print('dist=%s' % dist)
    # [END data]

    # [START model]
    print('Setting up model...')
    model = cp_model.CpModel()
    # [END model]

    # [START variables]
    print('Setting up variables...')
    x = []
    for f in range(len(facilities)):
        x.append(model.NewIntVar(0, 1, 'facility_%d' % f))
    y = []
    y_vector = []
    for f in range(len(facilities)):
        vector = []
        for c in range(len(customers)):
            vector.append(model.NewIntVar(0, 1, 'facility_%d_customer_%d' % (f, c)))
        y.append(vector)
        y_vector.extend(vector)
    y_transposed = transpose(y)
    # [END variables]

    # [START constraints]
    print('Setting up constraints...')
    print('Setting constraint 1...')
    # Constraint 1: Demand of the customers must not exceed the capacity of the facility
    for f in range(len(facilities)):
        model.Add(cp_model._ScalProd(y[f], demands) <= capacities[f])
    print('Setting constraint 2...')
    # Constraint 2: Each customer has to be served once by 1 facility
    for c in range(len(customers)):
        model.Add(sum(y_transposed[c]) == 1)
    print('Setting constraint 3...')
    # Constraint 3: Each customer can be assigned to only 1 facility
    for f in range(len(facilities)):
        for c in range(len(customers)):
            model.Add(y[f][c] <= x[f])
    # for f in range(len(facilities)):
    #    model.Add(cp_model._ScalProd(y[f], x[f]) == 1)
    # [END constraints]

    # [START objective function]
    print('Setting up objective function...')
    model.Minimize(cp_model._ScalProd(x, fixed_costs_factor))
    model.Minimize(cp_model._ScalProd(y_vector, dist_as_vector))
    # [END objective function]

    # [START solve]
    print('Solving...')
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 3600
    solver.parameters.log_search_progress = True
    # solver.parameters.num_search_workers = 2
    status = solver.Solve(model)
    # [END solve]

    cost = solver.ObjectiveValue()
    is_optimal = 0

    # [START solution]
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        is_optimal = 1 if status == cp_model.OPTIMAL else 0
        solution = []
        for f in range(len(facilities)):
            vector = [solver.Value(y[f][c]) for c in range(len(customers))]
            solution.append(vector)
        cost = calculate_cost_mix_integer_model(solution, cm, fixed_costs)
        used_facilities = calculate_used_facilities(solution)
        # cost = calculate_cost(facilities, customers, used_facilities)
    elif status == cp_model.INFEASIBLE:
        used_facilities = [0] * len(customers)
        cost = -1
        print('INFEASIBLE')
    else:
        used_facilities = [0] * len(customers)
        cost = -1
    # [END solution]

    return used_facilities, cost, is_optimal


def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(Facility(i - 1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3]))))

    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(Customer(i - 1 - facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    cm = CostMatrix(facilities, customers)

    if len(facilities) > 1000 or len(customers) > 3000:
        solution, cost, is_optimal = trivial_solution(facilities, customers)
    else:
        # Time Limit
        if len(facilities) <= 100 and len(customers) < 400:
            time_limit_seconds = 180
            num_threads = None
        elif len(facilities) <= 100:
            #time_limit_seconds = 600
            time_limit_seconds = 3600
            num_threads = 7
        elif len(facilities) <= 500:
            time_limit_seconds = 1200
            #time_limit_seconds = 3600 # Process is killed (Macbook 13 2020 i5 16Gb)
            num_threads = 7
        else
            #time_limit_seconds = 1200
            time_limit_seconds = 3600
            num_threads = None
        # solution, cost, is_optimal = trivial_solution(facilities, customers)
        # solution, cost, is_optimal = solve_ortools_cp_sat(cm, facilities, customers)
        solution, cost, is_optimal = solve_ortools_mip(cm, facilities, customers, time_limit_seconds,num_threads=num_threads)

    # prepare the solution in the specified output format
    output_data = '%.2f' % cost + ' ' + str(is_optimal) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def trivial_solution(facilities, customers):
    # build a trivial solution
    # pack the facilities one by one until all the customers are served
    solution = [-1] * len(customers)
    capacity_remaining = [f.capacity for f in facilities]

    facility_index = 0
    for customer in customers:
        if capacity_remaining[facility_index] >= customer.demand:
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand
        else:
            facility_index += 1
            assert capacity_remaining[facility_index] >= customer.demand
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand

    return solution, calculate_cost(facilities, customers, solution), 0


def calculate_cost_mix_integer_model(solution, cm, setup_cost):
    facility_cost = 0
    distance_cost = 0
    for f in range(len(solution)):
        facility_cost += setup_cost[f] if sum(solution[f]) >= 1 else 0
        for c in range(len(solution[0])):
            distance_cost += cm.get(f, c) if solution[f][c] == 1 else 0
    return facility_cost + distance_cost


def calculate_used_facilities(solution):
    customers_used_facility = [0] * len(solution[0])
    for f in range(len(solution)):
        for c in range(len(solution[0])):
            if solution[f][c] == 1:
                customers_used_facility[c] = f
    return customers_used_facility


def calculate_cost(facilities, customers, solution):
    used = [0] * len(facilities)
    for facility_index in solution:
        used[facility_index] = 1
    cost = sum([f.setup_cost * used[f.index] for f in facilities])
    for customer in customers:
        cost += length(customer.location, facilities[solution[customer.index]].location)
    return cost


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')
