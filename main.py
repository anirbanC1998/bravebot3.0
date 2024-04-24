import numpy as np
import pandas as pd


def initialize_grid(n=11, blocked_cells_count=10):
    grid = np.zeros((n, n))
    center = (n // 2, n // 2)
    grid[center] = 0  # Teleport pad

    # Having the diagonal cells in relation to the center to be blocked
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        grid[center[0] + dx, center[1] + dy] = -1

    flat_indices = np.arange(n * n)
    flat_indices = flat_indices[grid.flatten() != -1]
    blocked_indices = np.random.choice(flat_indices, size=blocked_cells_count, replace=False)
    grid[np.unravel_index(blocked_indices, (n, n))] = -1

    # Having the cells next to the teleport pad to be empty
    for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
        grid[center[0] + dx, center[1] + dy] = 0

    return grid, center


def initialize_positions(grid, center):
    n = len(grid)
    crew_position = (np.random.randint(n), np.random.randint(n))
    bot_position = (np.random.randint(n), np.random.randint(n))
    while grid[crew_position] == -1 or crew_position == center or crew_position == bot_position:
        crew_position = (np.random.randint(n), np.random.randint(n))
    while grid[bot_position] == -1 or bot_position == center or bot_position == crew_position:
        bot_position = (np.random.randint(n), np.random.randint(n))

    return crew_position, bot_position


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
                        # Set the transition probability initially for expected time
                        neighbor_index = x * n + y
                        P[current_index, neighbor_index] = prob
    return P


"""def update_probabilities_with_bot(grid, crew_pos, n):
    P = np.zeros((n * n, n * n))
    crew_index = crew_pos[0] * n + crew_pos[1]

    def is_valid(x, y):
        return 0 <= x < n and 0 <= y < n and grid[x][y] != -1

    # Normal random move in cardinal directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    possible_moves = [(crew_pos[0] + dx, crew_pos[1] + dy) for dx, dy in directions if
                      is_valid(crew_pos[0] + dx, crew_pos[1] + dy)]
    prob = 1 / len(possible_moves) if possible_moves else 0
    for move in possible_moves:
        neighbor_index = move[0] * n + move[1]
        P[crew_index, neighbor_index] = prob

    return P"""


def solve_for_expected_times(P, grid, center):
    n = len(grid)
    num_cells = n * n  # n^2 for every cell in the grid in terms of possibilities
    A = np.zeros((num_cells, num_cells))
    b = np.ones(num_cells)

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            if grid[i, j] == -1:  # Check if the cell is blocked
                A[idx, idx] = 1  # Set the diagonal to 1 to keep the equation consistent
                b[idx] = 0  # Set the corresponding b value to 0 for blocked cells
            elif (i, j) == center:
                A[idx, :] = 0  # Set entire row to 0
                A[idx, idx] = 1  # Set the diagonal to 1
                b[idx] = 0  # Expected time to reach the pad from the pad is 0
            else:
                A[idx, idx] = 1  # Set the diagonal element
                for (x, y) in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if 0 <= x < n and 0 <= y < n and grid[x, y] != -1:
                        neighbor_idx = x * n + y
                        A[idx, neighbor_idx] = -P[idx, neighbor_idx]

    T = np.linalg.solve(A, b)
    return T.reshape((n, n))


def initialize_reward_and_value_functions(grid, center):
    n = len(grid)
    num_states = n * n
    R = np.zeros((num_states, num_states))  # Reward matrix for all bot and crew combinations
    V = np.zeros((num_states, num_states))  # Value function matrix for all bot and crew combinations

    for bot_i in range(n):
        for bot_j in range(n):
            if grid[bot_i, bot_j] != -1:  # Bot must be in a valid cell
                bot_index = bot_i * n + bot_j
                for crew_i in range(n):
                    for crew_j in range(n):
                        if grid[crew_i, crew_j] != -1:  # Crew must also be in a valid cell
                            crew_index = crew_i * n + crew_j
                            # Distance to teleport pad, inverse the relation to have highest reward near center
                            distance = abs(crew_i - center[0]) + abs(crew_j - center[1])
                            R[bot_index, crew_index] = (2 * (n - 1)) - distance

    # Initialize with a high but finite initial cost instead of inf
    V = np.full((num_states, num_states), 1e3)  # Large number but not inf
    # Set known states directly related to the goal (e.g., crew at teleport pad) to zero
    for bot_index in range(num_states):
        teleport_index = center[0] * n + center[1]
        V[bot_index, teleport_index] = 0
    return R, V


def value_iteration(grid, reward, value, n, gamma=0.75, threshold=0.1):
    num_states = n * n
    delta = float('inf')

    while delta >= threshold:
        delta = 0
        for bot_index in range(num_states):
            if grid[bot_index // n][bot_index % n] == -1:
                continue  # Skip calculations for bot positions in blocked cells

            for crew_index in range(num_states):
                if grid[crew_index // n][crew_index % n] == -1:
                    continue  # Skip calculations for crew positions in blocked cells

                v = value[bot_index, crew_index]
                max_value = float('-inf')
                bot_actions = get_valid_actions(bot_index, grid, n)  # Retrieve valid actions for the bot

                for action in bot_actions:
                    expected_value = 0
                    next_bot_states = get_next_states(bot_index, action, grid, n)

                    for next_bot_index in next_bot_states:
                        crew_actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Crew can move in four cardinal directions
                        for ca in crew_actions:
                            next_crew_states = get_next_states(crew_index, ca, grid, n)
                            for next_crew_index in next_crew_states:
                                transition_prob = 1 / len(crew_actions) if len(crew_actions) > 0 else 0
                                expected_value += transition_prob * value[next_bot_index, next_crew_index]
                    action_value = (reward[bot_index, crew_index] + gamma * expected_value)
                    max_value = max(max_value, action_value)

                new_v = max_value if max_value != float('-inf') else v
                value[bot_index, crew_index] = new_v
                delta = max(delta, abs(v - new_v))
                print(f"Delta: {delta}")
                print(f"Value: {value[bot_index, crew_index]}")

    return value


def get_valid_actions(index, grid, n):
    x, y = index // n, index % n
    actions = []
    # Allow diagonal movements for the bot
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < n and 0 <= new_y < n and grid[new_x][new_y] != -1:
            actions.append((dx, dy))
    return actions


def get_next_states(index, action, grid, n):
    x, y = index // n, index % n
    dx, dy = action
    new_x, new_y = x + dx, y + dy
    if 0 <= new_x < n and 0 <= new_y < n and grid[new_x][new_y] != -1:
        return [new_x * n + new_y]  # Return a list of one next state
    return [index]  # Return the current state in a list if no valid move


def print_grid(grid, crew_position, bot_position):
    display_grid = np.array([' ' if x == 0 else 'X' for x in grid.flatten()]).reshape(grid.shape)
    display_grid[crew_position] = 'C'  # Mark the crew's current position
    display_grid[bot_position] = 'B'  # Mark the Bot's current position
    center = (len(grid) // 2, len(grid) // 2)
    display_grid[center] = 'T'  # Teleport pad
    df = pd.DataFrame(display_grid, index=[f"{i}" for i in range(grid.shape[0])],
                      columns=[f"{j}" for j in range(grid.shape[1])])
    print(df)
    print()


def decide_bot_move(grid, bot_position, crew_position, V, n):
    actions = [(0,0),(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1),
               (-1, -1)]
    min_time = float('inf')
    best_move = bot_position

    def is_valid(x, y):
        return 0 <= x < n and 0 <= y < n and grid[x, y] != -1

    # Iterate through each possible action
    for dx, dy in actions:
        new_bot_x, new_bot_y = bot_position[0] + dx, bot_position[1] + dy
        if is_valid(new_bot_x, new_bot_y):
            # Calculate the expected time for this move
            expected_time = 0
            # Assuming equal probability for each crew move initially (you'll adjust based on actual probabilities)
            possible_crew_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dcx, dcy in possible_crew_moves:
                new_crew_x, new_crew_y = crew_position[0] + dcx, crew_position[1] + dcy
                if is_valid(new_crew_x, new_crew_y):
                    crew_index = new_crew_x * n + new_crew_y
                    bot_index = new_bot_x * n + new_bot_y
                    probability = 1 / len(possible_crew_moves)  # Simplified uniform probability
                    expected_time += probability * V[bot_index, crew_index]

            # Add time step
            total_time = 1 + expected_time
            print(f"Total Time: {total_time}, Min Time: {min_time}")
            # Update the best move if this one is better
            if total_time < min_time:
                min_time = total_time
                best_move = (new_bot_x, new_bot_y)
                print(f"Picked a move: {best_move}")

    return best_move


def simulate_bot_crew_movement(grid, center, bot_toggle=True):
    n = len(grid)
    # Random initial positions for crew and bot, ensuring they are valid
    crew_position, bot_position = initialize_positions(grid, center)

    # Initialize reward and value functions
    R, V = initialize_reward_and_value_functions(grid, center)

    # Compute the optimal policies using value iteration
    # P = initialize_probabilities(grid)
    V = value_iteration(grid, R, V, n)

    steps = 0
    path = [crew_position]
    print("Crew starts at:", crew_position)
    print("Bot starts at:", bot_position)
    print_grid(grid, crew_position, bot_position)

    def is_valid(x, y):
        return 0 <= x < n and 0 <= y < n and grid[x][y] != -1

    while crew_position != center and steps < 10000:
        if bot_toggle:
            new_bot_position = decide_bot_move(grid, bot_position, crew_position, V, n)
        else:
            new_bot_position = bot_position  # Bot stays put if inactive

        # Determine if the bot is adjacent to the crew
        if abs(crew_position[0] - new_bot_position[0]) + abs(crew_position[1] - new_bot_position[1]) == 1:
            # Calculate the best move to maximize distance from bot, fleeing behavior
            possible_moves = [(dx, dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                              if is_valid(crew_position[0] + dx, crew_position[1] + dy)]
            if possible_moves:
                distances = {move: abs(move[0] + crew_position[0] - new_bot_position[0]) +
                                   abs(move[1] + crew_position[1] - new_bot_position[1])
                             for move in possible_moves}
                max_distance = max(distances.values())
                best_moves = [move for move, dist in distances.items() if dist == max_distance]
                move_choice = best_moves[np.random.choice(len(best_moves))]
        else:
            # Normal random move in cardinal directions
            possible_moves = [(dx, dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                              if is_valid(crew_position[0] + dx, crew_position[1] + dy)]
            move_choice = possible_moves[np.random.choice(len(possible_moves))] if possible_moves else (0, 0)

        new_crew_position = (crew_position[0] + move_choice[0], crew_position[1] + move_choice[1])

        # Update positions
        bot_position = new_bot_position
        crew_position = new_crew_position
        path.append(crew_position)

        # print_grid(grid, crew_position, bot_position)  # For debugging each step
        steps += 1
        print("Step", steps, "Crew at", crew_position, "Bot at", bot_position)
        if steps % 25 == 0 or crew_position == center:  # Print every 10 steps and at the end
            print("Step", steps, "Crew at", crew_position, "Bot at", bot_position)
            # print_grid(grid, crew_position, bot_position)

        if crew_position == center:
            print("Crew reaches the teleport pad at step", steps)
            print_grid(grid, crew_position, bot_position)
            break

    return path, steps


# Intialize crew only for Optimal Bot simulation
def initialize_crew_position(grid, center):
    n = len(grid)
    crew_position = (np.random.randint(n), np.random.randint(n))

    while grid[crew_position] == -1 or crew_position == center:
        crew_position = (np.random.randint(n), np.random.randint(n))

    return crew_position


def simulate_optimal_bot_crew_movement(grid, bot_position, center, V):
    # Random initial positions for crew ensuring they are valid
    n = len(grid)
    crew_position = initialize_crew_position(grid, center)

    steps = 0

    def is_valid(x, y):
        return 0 <= x < n and 0 <= y < n and grid[x][y] != -1

    while crew_position != center and steps < 10000:
        new_bot_position = decide_bot_move(grid, bot_position, crew_position, V, n)

        # Determine if the bot is adjacent to the crew
        if abs(crew_position[0] - new_bot_position[0]) + abs(crew_position[1] - new_bot_position[1]) == 1:
            # Calculate the best move to maximize distance from bot, fleeing behavior
            possible_moves = [(dx, dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                              if is_valid(crew_position[0] + dx, crew_position[1] + dy)]
            if possible_moves:
                distances = {move: abs(move[0] + crew_position[0] - new_bot_position[0]) +
                                   abs(move[1] + crew_position[1] - new_bot_position[1])
                             for move in possible_moves}
                max_distance = max(distances.values())
                best_moves = [move for move, dist in distances.items() if dist == max_distance]
                move_choice = best_moves[np.random.choice(len(best_moves))]
        else:
            # Normal random move in cardinal directions
            possible_moves = [(dx, dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                              if is_valid(crew_position[0] + dx, crew_position[1] + dy)]
            move_choice = possible_moves[np.random.choice(len(possible_moves))] if possible_moves else (0, 0)

        new_crew_position = (crew_position[0] + move_choice[0], crew_position[1] + move_choice[1])

        # Update positions
        bot_position = new_bot_position
        crew_position = new_crew_position
        steps += 1

    return crew_position, steps


def main_simulation():
    grid, center = initialize_grid()
    P_no_bot = initialize_probabilities(grid)
    mdv_transition_matrix = solve_for_expected_times(P_no_bot, grid, center)

    # Using Pandas DataFrame for better print
    df = pd.DataFrame(mdv_transition_matrix, index=[f"{i}" for i in range(mdv_transition_matrix.shape[0])],
                      columns=[f"{j}" for j in range(mdv_transition_matrix.shape[1])])
    print("Expected times to reach the teleport pad:")
    print(df)

    # Simulate optimal Bot movement
    bot_toggle = True
    path, steps = simulate_bot_crew_movement(grid, center, bot_toggle)
    print(f"Path : {path}, Steps: {steps}")


def optimal_simulation():
    # Initialize probabilities for no bot scenario to compare
    grid, center = initialize_grid()
    P_no_bot = initialize_probabilities(grid)
    mdv_transition_matrix = solve_for_expected_times(P_no_bot, grid, center)

    # Using Pandas DataFrame for better print
    df = pd.DataFrame(mdv_transition_matrix, index=[f"{i}" for i in range(mdv_transition_matrix.shape[0])],
                      columns=[f"{j}" for j in range(mdv_transition_matrix.shape[1])])
    print("Expected times to reach the teleport pad:")
    print(df)

    n = len(grid)
    best_position = None
    best_time = float('inf')
    results = {}

    R, V = initialize_reward_and_value_functions(grid, center)
    V = value_iteration(grid, R, V, n)

    # Test each valid position on the grid
    for i in range(n):
        for j in range(n):
            if grid[i][j] != -1:  # Ensure the bot does not start in a blocked cell
                bot_position = (i, j)
                # print(f"Testing bot start position at {bot_position}")
                _, time_taken = simulate_optimal_bot_crew_movement(grid, bot_position, center, V)
                results[bot_position] = time_taken
                if time_taken < best_time:
                    best_time = time_taken
                    best_position = bot_position

    print(f"Optimal bot position is {best_position} with expected time {best_time}")
    return best_position, results


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)  # Ensures all rows are displayed
    pd.set_option('display.max_columns', None)  # Ensures all columns are displayed
    pd.set_option('display.width', None)  # Adjusts the display width for wide DataFrames

    np.random.seed(0)  # Fixing Ship Layout, Crew, and Bot Position
    # 0, 3, 5, 12 are good fixed layouts

    # Basic simlulation using fixed Crew and fixed Bot placement
    main_simulation()

    # Simulate the optimal Bot placement
    # optimal_simulation()
