import java.io.*;
import java.util.*;

/**
 * The class <code>Solver</code> is an implementation of a greedy algorithm to solve the knapsack problem.
 */
public class Solver {

    /**
     * The main class
     */
    public static void main(String[] args) {
        try {
            solve(args);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // ++++++++++++++++++++++++++++++++++
    // STATIC CLASSES
    // ++++++++++++++++++++++++++++++++++

    // Keep items original order
    public static class Item {
        private final int index;
        private final int value;
        private final int weight;
        private final double valuePerWeightUnit;

        public Item(int index, int value, int weight) {
            this.index = index;
            this.value = value;
            this.weight = weight;
            this.valuePerWeightUnit = ((double) value) / ((double) weight);
        }

        public int getIndex() {
            return index;
        }

        public int getValue() {
            return value;
        }

        public int getWeight() {
            return weight;
        }

        public double getValuePerWeightUnit() {
            return valuePerWeightUnit;
        }

        @Override
        public String toString() {
            return "Item{" +
                    "index=" + index +
                    ", value=" + value +
                    ", weight=" + weight +
                    ", valuePerWeightUnit=" + valuePerWeightUnit +
                    '}';
        }
    }

    public static class Solution {
        private final int weight;
        private final int value;
        private final int[] taken;

        public Solution(int weight, int value, int[] taken) {
            this.weight = weight;
            this.value = value;
            this.taken = taken;
        }

        public int getWeight() {
            return weight;
        }

        public int getValue() {
            return value;
        }

        public int[] getTaken() {
            return taken;
        }

        @Override
        public String toString() {
            return "Solution{" +
                    "weight=" + weight +
                    ", value=" + value +
                    ", taken=" + Arrays.toString(taken) +
                    '}';
        }
    }

    public static class Node {
        private final int value;
        private final int capacityLeft;
        private final double expectation;
        private final int[] taken;
        private final int pos;

        public Node(int value, int capacityLeft, double expectation, int[] taken, int pos) {
            this.value = value;
            this.capacityLeft = capacityLeft;
            this.expectation = expectation;
            this.taken = taken;
            this.pos = pos;
        }

        public int getValue() {
            return value;
        }

        public int getCapacityLeft() {
            return capacityLeft;
        }

        public double getExpectation() {
            return expectation;
        }

        public int[] getTaken() {
            return taken;
        }

        public int getPos() {
            return pos;
        }
    }

    /**
     * Read the instance, solve it, and print the solution in the standard output
     */
    public static void solve(String[] args) throws IOException {
        String fileName = null;

        // get the temp file name
        for (String arg : args) {
            if (arg.startsWith("-file=")) {
                fileName = arg.substring(6);
            }
        }
        if (fileName == null)
            return;

        // read the lines out of the file
        List<String> lines = new ArrayList<String>();

        BufferedReader input = new BufferedReader(new FileReader(fileName));
        try {
            String line = null;
            while ((line = input.readLine()) != null) {
                lines.add(line);
            }
        } finally {
            input.close();
        }


        // parse the data in the file
        String[] firstLine = lines.get(0).split("\\s+");
        int numItems = Integer.parseInt(firstLine[0]);
        int capacity = Integer.parseInt(firstLine[1]);

        ArrayList<Item> items = new ArrayList<>();

        for (int i = 1; i < numItems + 1; i++) {
            String line = lines.get(i);
            String[] parts = line.split("\\s+");
            items.add(new Item(i - 1, Integer.parseInt(parts[0]), Integer.parseInt(parts[1])));
        }

        //items.forEach(item -> System.out.println(item.toString()));

        Solution sol;

        //Solution sol = solveWithTrivialGreedyAlgorithm(items, capacity);
        //Solution sol = solveWithABranchAndBoundAlgorithm(items, capacity);
        // Solution sol = DPMemoizationKnapsack(items, capacity, items.size());
        //Solution sol = DPTopDownKnapsack(items, capacity);

        if (items.size() <= 200) {
            sol = DPTopDownKnapsackV2(items, capacity);
        } else {
            items.sort(Comparator.comparing(Item::getValuePerWeightUnit, Comparator.reverseOrder()).thenComparing(Item::getWeight));
            sol = solveWithABranchAndBoundAlgorithm(items, capacity);
        }

        // prepare the solution in the specified output format
        System.out.println(sol.getValue() + " 0");
        for (int i = 0; i < numItems; i++) {
            System.out.print(sol.getTaken()[i] + " ");
        }
        System.out.println("");
    }

    public static Solution solveWithTrivialGreedyAlgorithm(ArrayList<Item> items, int capacity) {
        // a trivial greedy algorithm for filling the knapsack
        // it takes items in-order until the knapsack is full
        int value = 0;
        int weight = 0;
        int[] taken = new int[items.size()];

        // >> Algorithm
        for (int i = 0; i < items.size(); i++) {
            Item item = items.get(i);
            if (weight + item.getWeight() <= capacity) {
                taken[item.index] = 1;
                value += item.getValue();
                weight += item.getWeight();
            } else {
                taken[item.index] = 0;
            }
        }
        // << Algorithm

        return new Solution(weight, value, taken);
    }

    // ++++++++++++++++++++++++++++++++++
    // BRANCH AND BOUND
    // ++++++++++++++++++++++++++++++++++

    /**
     * Max expected value from current capacity
     * Upper-bound
     */
    public static double getExpectationOld(final ArrayList<Item> items, int capacity, int curPos) {
        double expectation = 0.0;
        for (int i = curPos; i < items.size(); i++) { // Sort descendently by valuePerWeightUnit
            Item item = items.get(i);
            //if (capacity >= item.weight) {
            if (capacity >= item.weight) {
                expectation += item.value;
                capacity -= item.weight;
            } else {
                expectation += item.getValuePerWeightUnit() * capacity;
                break;
            }
        }
        return expectation;
    }

    public static double getExpectation(final ArrayList<Item> items, int capacity, int curPos) {
        double expectation = 0.0;
        int curCapacity = capacity;
        for (int i = curPos; i < items.size(); i++) { // Sort descendently by valuePerWeightUnit
            Item item = items.get(i);
            //if (curCapacity >= item.weight) {
            if (curCapacity >= item.weight) {
                expectation += item.value;
                curCapacity -= item.weight;
            } else {
                expectation += item.getValuePerWeightUnit() * curCapacity;
                break;
            }
        }
        return expectation;
    }

    // Not use recursion to prevent stack-overflow
    public static Solution solveWithABranchAndBoundAlgorithm(ArrayList<Item> items, int capacity) {

        int bestValue = 0;
        int[] bestTaken = new int[items.size()];
        Arrays.fill(bestTaken, 0);

        int startValue = 0;
        int startCapacity = capacity;
        double startExpectation = getExpectation(items, capacity, 0);
        int[] startTaken = new int[items.size()];
        Arrays.fill(startTaken, 0);
        int startPos = 0;

        Stack<Node> stack = new Stack<>();
        stack.push(new Node(startValue, startCapacity, startExpectation, startTaken, startPos));

        while (!stack.isEmpty()) {
            Node node = stack.pop();
            if (node.getCapacityLeft() < 0) continue; // Left capacity is not enough, then backtrack
            if (node.getExpectation() <= bestValue) continue; // Current expectation is worse than the best, then backtrack
            if (node.getValue() > bestValue) {
                // Better solution found
                bestValue = node.getValue();
                bestTaken = node.getTaken();
            }
            if (node.getPos() >= items.size())
                continue; // Leaf of the tree found (next item does not exist), then backtrack
            Item curItem = items.get(node.getPos());
            // Try not to take next item (left child branch)
            Node leftNode = new Node(
                    node.getValue(),
                    node.getCapacityLeft(),
                    node.getValue() + getExpectation(items, node.getCapacityLeft(), node.getPos() + 1),
                    node.getTaken(),
                    node.getPos() + 1
            );
            stack.push(leftNode);
            // Try to take next item (right child branch)
            int[] taken = Arrays.copyOfRange(node.getTaken(), 0, node.getTaken().length);
            int capacityLeft = node.getCapacityLeft() - curItem.getWeight();
            taken[curItem.getIndex()] = 1;
            Node rightNode = new Node(
                    node.getValue() + curItem.getValue(),
                    capacityLeft,
                    node.getValue() + getExpectation(items, capacityLeft, node.getPos() + 1),
                    taken,
                    node.getPos() + 1
            );
            stack.push(rightNode);
        }

        return new Solution(-1, bestValue, bestTaken);
    }

    // ++++++++++++++++++++++++++++++++++
    // DYNAMIC PROGRAMMING
    // ++++++++++++++++++++++++++++++++++

    public static int max(int a, int b) {
        return (a > b) ? a : b;
    }

    public static Solution DPRecursiveKnapsack(int items, int capacity) throws Exception {
        throw new Exception();
    }

    /*public static final int BOUND = 100000;

    public static int[][] memo = new int[BOUND][BOUND];

    public static Solution DPMemoizationKnapsack(ArrayList<Item> items, int capacity, int n) throws Exception {
        if (items.size() == 0 || capacity == 0) return 0;
        if (memo[n][capacity] != -1) return memo[n][capacity];
        if (items.get(n).getWeight() <= capacity)
            return memo[n][capacity] = max(
                    items.get(n - 1) + DPMemoizationKnapsack(items, capacity - items.get(n - 1).getWeight(), n - 1),
                    DPMemoizationKnapsack(items, capacity, n - 1)
            );
        else if (items.get(n - 1).getWeight() > capacity)
            return memo[n][capacity] = DPMemoizationKnapsack(item, capacity, n - 1);
        return 0;
    }*/

    public static Solution DPTopDownKnapsack(ArrayList<Item> items, int capacity) {
        int BOUND = 100000;
        int n = items.size();
        int[][] matrix = new int[BOUND][BOUND];

        // Top Down Initialization
        for (int i = 0; i < n + 1; i++) {
            for (int j = 0; j < n + 1; j++) {
                if (i == 0 || j == 0)
                    matrix[i][j] = 0;
            }
        }

        // Solve all sub-problems, using the results from the previous problems
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < n + 1; j++) {
                Item item = items.get(i - 1);
                // Choice diagram: replace n with i and w with j
                if (item.getWeight() <= capacity) {
                    matrix[i][j] = max(item.getValue() + matrix[i - 1][j - item.getWeight()], matrix[i - 1][j]);
                } else {
                    matrix[i][j] = matrix[i - 1][j];
                }
            }
        }

        System.out.println(matrix[n][capacity]);
        return new Solution(0, 0, new int[2]);
    }

    public static Solution DPTopDownKnapsackV2(ArrayList<Item> items, int capacity) {
        int n = items.size();
        int[] taken = new int[n];
        int[][] table = new int[capacity + 1][n + 1];

        // Top Down Initialization
        for (int i = 0; i < n + 1; i++) {
            for (int j = 0; j < capacity + 1; j++) {
                if (i == 0 || j == 0) {
                    table[j][i] = 0;
                    continue;
                }
                Item item = items.get(i - 1);
                if (item.getWeight() <= j) {
                    table[j][i] = max(table[j][i - 1], item.getValue() + table[j - item.getWeight()][i - 1]);
                } else {
                    table[j][i] = table[j][i - 1];
                }
            }
        }

        // Backtrack
        int j = capacity + 1;
        for (int i = n; i > 0; i--) {
            if (table[j - 1][i] > table[j - 1][i - 1]) {
                taken[i - 1] = 1;
                j = j - items.get(i - 1).getWeight();
            }
        }

        int value = table[capacity][n];
        return new Solution(0, value, taken);
    }
}
