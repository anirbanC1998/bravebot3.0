import numpy as np
import pandas as pd


def initialize_grid(n=11, blocked_cells_count=10):
    grid = np.zeros((n, n))
    center = (n // 2, n // 2)
    grid[center] = 0  # Teleport pad

    # having the diagonal cells in relation to the center to be blocked
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        grid[center[0] + dx, center[1] + dy] = -1

    np.random.seed(0)
    flat_indices = np.arange(n * n)
    flat_indices = flat_indices[grid.flatten() != -1]
    blocked_indices = np.random.choice(flat_indices, size=blocked_cells_count, replace=False)
    grid[np.unravel_index(blocked_indices, (n, n))] = -1

    return grid, center


def initialize_probabilities(grid):
    n = len(grid)
    P = np.zeros((n * n, n * n))

    def is_valid(x, y):
        return 0 <= x < n and 0 <= y < n and grid[x, y] != -1

    for i in range(n):
        for j in range(n):
            if grid[i, j] != -1:
                current_index = i * n + j
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                valid_neighbors = [idx for idx in neighbors if is_valid(*idx)]
                if valid_neighbors:
                    prob = 1 / len(valid_neighbors)
                    for (x, y) in valid_neighbors:
                        neighbor_index = x * n + y
                        P[current_index, neighbor_index] = prob
    return P


def solve_for_expected_times(P, grid, center):
    n = len(grid)
    num_cells = n * n
    A = np.zeros((num_cells, num_cells))
    b = np.ones(num_cells)  # The 1 in the equation: 1 + sum(...)

    for i in range(num_cells):
        A[i, i] = 1
        for j in range(num_cells):
            if P[i, j] > 0:
                A[i, j] -= P[i, j]

    center_index = center[0] * n + center[1]
    A[center_index, :] = 0
    A[center_index, center_index] = 1
    b[center_index] = 0

    T = np.linalg.solve(A, b)
    return T.reshape((n, n))


def simulate_crew_movement(grid, P, center):
    n = len(grid)
    current_position = (np.random.randint(n), np.random.randint(n))
    while grid[current_position] == -1 or current_position == center:
        current_position = (np.random.randint(n), np.random.randint(n))

    steps = 0
    path = [current_position]
    print("Crew starts at:", current_position)

    while current_position != center:
        current_index = current_position[0] * n + current_position[1]
        probabilities = P[current_index]
        next_index = np.random.choice(np.arange(n * n), p=probabilities)
        next_position = (next_index // n, next_index % n)

        current_position = next_position
        path.append(current_position)
        steps += 1

        if steps % 10 == 0 or current_position == center:  # Print every 10 steps and at the end
            print("Step", steps, "at", current_position)

        if current_position == center:
            print("Crew reaches the teleport pad at step", steps)
            break

    return path, steps


def main():
    pd.set_option('display.max_rows', None)  # Ensures all rows are displayed
    pd.set_option('display.max_columns', None)  # Ensures all columns are displayed
    pd.set_option('display.width', None)  # Adjusts the display width for wide DataFrames

    grid, center = initialize_grid()
    P = initialize_probabilities(grid)
    T_no_bot = solve_for_expected_times(P, grid, center)

    # Using Pandas DataFrame for pretty printing
    df = pd.DataFrame(T_no_bot, index=[f"Row {i}" for i in range(T_no_bot.shape[0])],
                      columns=[f"Col {j}" for j in range(T_no_bot.shape[1])])
    print("Expected times to reach the teleport pad:")
    print(df)

    path, steps = simulate_crew_movement(grid, P, center)


if __name__ == "__main__":
    main()
