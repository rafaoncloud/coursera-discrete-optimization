# #!/bin/bash

# Example: bash run.sh ks_10000_0

javac Solver.java | java -Xmx12g -Xms200m Solver -file=data/$1
