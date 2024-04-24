import numpy as np
import pandas as pd



def initialize_grid(n=11, blocked_cells_count=10):
    grid = np.zeros((n, n))
    center = (n // 2, n // 2)
    grid[center] = 0  # Teleport pad

    #Having the diagonal cells in relation to the center to be blocked
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        grid[center[0] + dx, center[1] + dy] = -1


    flat_indices = np.arange(n * n)
    flat_indices = flat_indices[grid.flatten() != -1]
    blocked_indices = np.random.choice(flat_indices, size=blocked_cells_count, replace=False)
    grid[np.unravel_index(blocked_indices, (n, n))] = -1
    
    #Having the cells next to the teleport pad to be empty
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
    for i in range(n):
        for j in range(n):
            if grid[i, j] != -1:
                current_index = i * n + j
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                valid_neighbors = [x for x in neighbors if 0 <= x[0] < n and 0 <= x[1] < n and grid[x[0], x[1]] != -1]
                for neighbor in valid_neighbors:
                    neighbor_index = neighbor[0] * n + neighbor[1]
                    P[current_index, neighbor_index] = 1 / len(valid_neighbors)
    return P

def update_probabilities_with_bot(grid, crew_pos, n):
    P = np.zeros((n * n, n * n))
    crew_index = crew_pos[0] * n + crew_pos[1]

    # Set default move probabilities
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_pos = (crew_pos[0] + dx, crew_pos[1] + dy)
        if 0 <= new_pos[0] < n and 0 <= new_pos[1] < n and grid[new_pos[0]][new_pos[1]] != -1:
            neighbor_index = new_pos[0] * n + new_pos[1]
            P[crew_index, neighbor_index] = 1  # Equal probability initially

    # Normalize probabilities to ensure they sum to 1
    total = np.sum(P[crew_index, :])
    if total > 0:
        P[crew_index, :] /= total

    return P

def solve_for_expected_times(P, grid, center):
    n = len(grid)
    num_cells = n * n # n^2 for every cell in the grid in terms of possibilities
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
                for (x, y) in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                    if 0 <= x < n and 0 <= y < n and grid[x, y] != -1:
                        neighbor_idx = x * n + y
                        A[idx, neighbor_idx] = -P[idx, neighbor_idx]

    T = np.linalg.solve(A, b) 
    return T.reshape((n, n))

def initialize_reward_and_value_functions(grid, center):
    n = len(grid)
    num_states = n * n
    R = np.zeros((num_states, num_states))
    V = np.zeros(num_states)
    
    for i in range(n):
        for j in range(n):
            if grid[i, j] != -1:
                current_index = i * n + j
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]:
                    new_i, new_j = i + dx, j + dy
                    if 0 <= new_i < n and 0 <= new_j < n and grid[new_i, new_j] != -1:
                        new_index = new_i * n + new_j
                        # Negative reward for greater distance to center
                        distance = abs(new_i - center[0]) + abs(new_j - center[1])
                        R[current_index, new_index] = distance
    return R, V

def value_iteration(R, P, V, n, gamma=0.90, threshold=0.01):
    num_states = n * n
    delta = float('inf')
    while delta > threshold:
        delta = 0
        for s in range(num_states):
            v = V[s]
            max_value = float('-inf')
            for s_prime in range(num_states):  # Explore all possible states s_prime
                if P[s, s_prime] > 0:  # Ensure there is a possible transition
                    action_value = P[s, s_prime] * (R[s, s_prime] + gamma * V[s_prime])
                    max_value = max(max_value, action_value)
            new_v = max_value if max_value != float('-inf') else v
            V[s] = new_v
            delta = max(delta, abs(v - new_v))
    return V

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

def decide_bot_move(grid, bot_position, crew_position, V, n, bot_moves):
    best_bot_move = None
    min_time_to_center = float('inf')

    # Evaluate each possible bot move
    for move in bot_moves:
        new_bot_pos = (bot_position[0] + move[0], bot_position[1] + move[1])
        if 0 <= new_bot_pos[0] < n and 0 <= new_bot_pos[1] < n and grid[new_bot_pos[0]][new_bot_pos[1]] != -1:
            # Assume bot influences crew member's next move
            P_temp = update_probabilities_with_bot(grid, crew_position, new_bot_pos, n)
            expected_time = 0

            # Aggregate potential outcomes from this new bot position
            for i in range(n):
                for j in range(n):
                    idx = i * n + j
                    expected_time += P_temp[crew_position[0] * n + crew_position[1], idx] * V[idx]

            # Select the new bot position if it minimizes the expected time
            if expected_time < min_time_to_center:
                min_time_to_center = expected_time
                best_bot_move = new_bot_pos

    return best_bot_move if best_bot_move else bot_position


def get_valid_bot_moves(bot_position, grid, n):
    valid_moves = []
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dx, dy in directions:
        new_x, new_y = bot_position[0] + dx, bot_position[1] + dy
        if 0 <= new_x < n and 0 <= new_y < n and grid[new_x][new_y] != -1:
            valid_moves.append((new_x, new_y))
    return valid_moves


def simulate_bot_crew_movement(grid, center, bot_toggle = True):
    n = len(grid)
    # Random initial positions for crew and bot, ensuring they are valid
    crew_position, bot_position = initialize_positions(grid, center)
    
    # Initialize reward and value functions
    R, V = initialize_reward_and_value_functions(grid, center)
    
    # Compute the optimal policies using value iteration
    P = initialize_probabilities(grid)  # Make sure this is defined correctly
    V = value_iteration(R, P, V, n)
    
    steps = 0
    path = [crew_position]
    print("Crew starts at:", crew_position)
    print("Bot starts at:", bot_position)
    print_grid(grid, crew_position, bot_position)


    while crew_position != center and steps < 10000:
        if bot_toggle:
            bot_moves = get_valid_bot_moves(bot_position, grid, n)
            new_bot_position = decide_bot_move(grid, bot_position, crew_position, V, n, bot_moves)
        else:
            new_bot_position = bot_position
            
        # Calculate crew's next move based on updated probabilities
        P_with_bot = update_probabilities_with_bot(grid, crew_position, new_bot_position, n)
        crew_probabilities = P_with_bot[crew_position[0] * n + crew_position[1]]
        next_index = np.random.choice(np.arange(n * n), p=crew_probabilities)
        new_crew_position = (next_index // n, next_index % n)

        if abs(crew_position[0] - new_bot_position[0]) <= 1 and abs(crew_position[1] - new_bot_position[1]) <= 1:
            # Flee logic when bot is adjacent in any direction
            best_move = crew_position
            max_distance = -1
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Only cardinal directions
                new_pos = (crew_position[0] + dx, crew_position[1] + dy)
                if 0 <= new_pos[0] < n and 0 <= new_pos[1] < n and grid[new_pos[0]][new_pos[1]] != -1:
                    distance = abs(bot_position[0] - new_pos[0]) + abs(bot_position[1] - new_pos[1])
                    if distance > max_distance:
                        max_distance = distance
                        best_move = new_pos
            new_crew_position = best_move

        bot_position = new_bot_position
        crew_position = new_crew_position
        path.append(crew_position)
        
        #print_grid(grid, crew_position, bot_position)  # For debugging each step
        steps += 1
        print("Step", steps, "Crew at", crew_position, "Bot at", bot_position)
        
        if steps % 25 == 0 or crew_position == center:  # Print every 10 steps and at the end
            print("Step", steps, "Crew at", crew_position, "Bot at", bot_position)
            #print_grid(grid, crew_position, bot_position)

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
    while crew_position != center and steps < 10000:
        # Update probabilities with bot influence
        P_with_bot = update_probabilities_with_bot(grid, crew_position, bot_position, n)
        
        # Calculate crew's next move based on updated probabilities
        crew_probabilities = P_with_bot[crew_position[0] * n + crew_position[1]]
        next_index = np.random.choice(np.arange(n * n), p=crew_probabilities)
        new_crew_position = (next_index // n, next_index % n)
        
        # Move bot towards the best next position
        new_bot_position = decide_bot_move(grid, bot_position, crew_position, V, n)
        
        if abs(crew_position[0] - new_bot_position[0]) + abs(crew_position[1] - new_bot_position[1]) == 1:
            # Calculate the best move to maximize distance from bot
            possible_moves = [(dx, dy) for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                              if 0 <= crew_position[0] + dx < n and 0 <= crew_position[1] + dy < n and
                              grid[crew_position[0] + dx][crew_position[1] + dy] != -1]
            if possible_moves:
                distances = {move: abs(move[0] + crew_position[0] - new_bot_position[0]) + 
                                    abs(move[1] + crew_position[1] - new_bot_position[1])
                             for move in possible_moves}
                max_distance = max(distances.values())
                best_moves = [move for move, dist in distances.items() if dist == max_distance]
                best_indices = [possible_moves.index(move) for move in best_moves]
                chosen_index = np.random.choice(best_indices)  
                move_choice = possible_moves[chosen_index]
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
    V = value_iteration(grid, R, P_no_bot, V, n)

    # Test each valid position on the grid
    for i in range(n):
        for j in range(n):
            if grid[i][j] != -1:  # Ensure the bot does not start in a blocked cell
                bot_position = (i, j)
                #print(f"Testing bot start position at {bot_position}")
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
    
    np.random.seed(0) # Fixing Ship Layout, Crew, and Bot Position
    #0, 3, 5, 12 are good fixed layouts
    
    # Basic simlulation using fixed Crew and fixed Bot placement
    main_simulation()
    
    # Simulate the optimal Bot placement
    #optimal_simulation()

