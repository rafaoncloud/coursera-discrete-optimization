## Facility Location Problem

### (Naive) MIP Model Formulation

#### Data

N         = Number of facilities

M         = Number of customers

d(c)      = Demand of each customer c

cap(f)    = Capacity of each facility f

s(f)      = Fixed cost of each facility c

dist(f,c) = Distance cost from each facility f to customer c

#### Decision Variables

x(f)   = 1 if facility f is assigned, 0 otherwise

y(f,c) = 1 if customer assigned to facility f, 0 otherwise

#### Objective Function & Constraints

min f(x) = \sum_{f=1}^{N} x_(f).s(f) + \sum_{f=1}^{N} \sum_{c=1}^{M} y(f,c).dist(f,c)

s.t.

\sum_{c=1}^{M} y(f,c).d(f,c) <= cap(f) , \forall f \in N (Demand of the customers must not exceed the capacity of the facility)

\sum_{f=1}^{N} y(f,c) = 1, \forall c \in M (Each customer has to be served once by 1 facility)

\sum_{f=1}^{N} y(f,c) = 1 <= x(f), \forall c \in M (Each customer can be assigned to only 1 facility)

