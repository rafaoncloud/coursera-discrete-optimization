#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import random
import math
import time
from datetime import datetime
from copy import deepcopy

random.seed(123)


# ++++++++++++++++++++++++++
# PROBLEM
# ++++++++++++++++++++++++++

class Location:
    def __init__(self, index, x, y):
        self.index = index
        self.x = x
        self.y = y


class Customer:
    def __init__(self, index, demand, location):
        self.index = index
        self.demand = demand
        self.location = location


class CostMatrix:
    def __init__(self, customers):
        self.cost_matrix = [[0 for _ in customers] for _ in customers]
        # First customer is the depot location
        for c_from in customers:
            for c_to in customers:
                self.cost_matrix[c_from.location.index][c_to.location.index] = length(c_from.location, c_to.location)

    def get(self, c1, c2):
        # Distance from -> to location given customers
        return self.getl(c1.location, c2.location)

    def getl(self, l1, l2):
        # Distance from -> to location
        return self.cost_matrix[l1.index][l2.index]


def length(l1, l2):
    # Euclidean Distance
    return math.sqrt((l1.x - l2.x) ** 2 + (l1.y - l2.y) ** 2)


class Problem:
    def __init__(self, cost_matrix: CostMatrix, depot, customers, num_vehicles, vehicle_capacity):
        self.cm = cost_matrix
        self.depot = depot
        self.customers = customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity


# ++++++++++++++++++++++++++
# ALGORITHM - Simulated Annealing
# ++++++++++++++++++++++++++

def search(p: Problem, routes, time_limit=60, max_iterations=1000000):
    current_solution = deepcopy(routes)
    current_cost = solution_cost(p, current_solution)
    best_solution = deepcopy(current_solution)
    best_cost = solution_cost(p, best_solution)
    restart = 0
    iteration = 0
    INITIAL_TEMPERATURE = len(p.customers) / 2
    T = INITIAL_TEMPERATURE
    alpha = 0.9999
    min_T = math.pow(1, 6)
    start = time.time()
    diff = time.time() - start
    print('Local search started...')
    while diff < time_limit:
        print('[Iteration=%d/%d][Timer=%.3f/%.3f][CurCost=%f][BestCost=%f][T=%f][Restarts=%d]' %
              (iteration + 1, max_iterations, diff, time_limit, current_cost, best_cost, T, restart))
        if T <= min_T:
            T = INITIAL_TEMPERATURE
            restart += 1
        neighbor_solution = get_neighbor(p, current_solution)
        if neighbor_solution is not None:
            if accept(p, current_solution, current_cost, neighbor_solution, T):
                current_solution = neighbor_solution
                current_cost = solution_cost(p, current_solution)
                if current_cost < best_cost:
                    best_solution = deepcopy(current_solution)
                    best_cost = current_cost
        iteration += 1
        T *= alpha
        diff = time.time() - start
    print('Search ended after %d iterations and %d restarts at %.3f' % (iteration, restart, time.time() - start))
    return best_solution


def accept(p, current_solution, current_solution_cost, neighbor_solution, T):
    old_cost = current_solution_cost
    new_cost = solution_cost(p, neighbor_solution)
    delta = new_cost - old_cost
    if new_cost <= old_cost: return True
    if math.exp(-abs(delta) / T) > random.random(): return True
    return False


def get_neighbor(p: Problem, solution):
    neighbor = deepcopy(solution)
    v1 = random.randint(0, len(solution)-1)  # from what vehicle
    while len(solution[v1]) == 0:
        v1 = random.randint(0, len(solution)-1)
    c1 = random.randint(0, len(solution[v1])-1)  # what customer
    tmp = neighbor[v1][c1]
    cap1, cap2 = p.vehicle_capacity+1, 0
    while cap1 > p.vehicle_capacity or cap2 < tmp.demand:
        v2 = random.randint(0, len(solution)-1)  # to what vehicle
        c2 = random.randint(0, len(solution[v2]))  # in what position
        cap1 = sum(solution[v1][x].demand for x in range(len(solution[v1])) if x!=c1)
        if c2 < len(solution[v2]):
             cap1 += solution[v2][c2].demand
        cap2 = p.vehicle_capacity - sum(solution[v2][x].demand for x in range(len(solution[v2])) if x!=c2)
    if c2 < len(solution[v2]):
        neighbor[v1][c1] = neighbor[v2][c2]
        neighbor[v2][c2] = tmp
    else:
        neighbor[v1].remove(tmp)
        neighbor[v2].insert(c2, tmp)
    return neighbor

# ++++++++++++++++++++++++++
# START
# ++++++++++++++++++++++++++


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    num_customers = int(parts[0])
    num_vehicles = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    for i in range(1, num_customers + 1):
        line = lines[i]
        parts = line.split()
        index = i - 1
        location = Location(index, float(parts[1]), float(parts[2]))
        customers.append(Customer(index, int(parts[0]), location))

    # the depot is always the first customer in the input
    depot = customers[0]

    cm = CostMatrix(customers)
    problem = Problem(cm, depot, customers, num_vehicles, vehicle_capacity)

    # Initial Solution
    routes = trivial_solution(problem)
    # Solve from initial solution
    routes = search(problem, routes, time_limit=600)

    # checks that the number of customers served is correct
    assert sum([len(r) for r in routes]) == len(customers) - 1

    cost = solution_cost(problem, routes)

    # prepare the solution in the specified output format
    outputData = '%.2f' % cost + ' ' + str(0) + '\n'
    for v in range(0, num_vehicles):
        outputData += str(depot.index) + ' ' + ' '.join(
            [str(customer.index) for customer in routes[v]]) + ' ' + str(depot.index) + '\n'

    return outputData


def trivial_solution(p: Problem):
    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    routes = []

    remaining_customers = set(p.customers)
    remaining_customers.remove(p.depot)

    for v in range(0, p.num_vehicles):
        # print "Start Vehicle: ",v
        routes.append([])
        capacity_remaining = p.vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
            used = set()
            order = sorted(remaining_customers,
                           key=lambda customer: -customer.demand * len(p.customers) + customer.index)
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    routes[v].append(customer)
                    # print '   add', ci, capacity_remaining
                    used.add(customer)
            remaining_customers -= used

    return routes


def solution_cost(p: Problem, routes):
    # calculate the cost of the solution; for each vehicle the length of the route
    cost = 0
    for v in range(0, p.num_vehicles):
        vehicle_route = routes[v]
        if len(vehicle_route) > 0:
            cost += p.cm.get(p.depot, vehicle_route[0])  # (START) From depot to first activity
            for i in range(0, len(vehicle_route) - 1):
                cost += p.cm.get(vehicle_route[i], vehicle_route[i + 1])
            cost += p.cm.get(vehicle_route[-1], p.depot)  # (END) From last activity to depot
    return cost


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')
