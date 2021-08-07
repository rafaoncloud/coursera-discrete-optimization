# Set cover

n items (regions)
m sets (fire stations)
c_{i} set costs
covered items S_{i}
x_{i} = 1 if set i is selected

min sum_{i}^{M} c_{i} * x_{i}
s.t.
sum_{i \in M} (j \in S_{i}) x_{i} >= 1 (j \in N) // All regions have to be covered
x_{i} \in {0, 1}

// Input Data
N M // First line - Num items and sets
c_{M - 1} s_{M - 1}_0 ... s_{M - 1}_$ // Remanining lines - First value is the cost, then each value is the index of the item (region) covered


// Output
obj opt // Objective value and is optimal?
x_0 x_1 ... x_{M - 1} // Values of the decision variables


