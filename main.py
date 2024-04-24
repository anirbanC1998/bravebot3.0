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

def update_probabilities_with_bot(grid, crew_pos, n):
    P = np.zeros((n * n, n * n))
    crew_index = crew_pos[0] * n + crew_pos[1]

    def is_valid(x, y):
        return 0 <= x < n and 0 <= y < n and grid[x][y] != -1

    # Normal random move in cardinal directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    possible_moves = [(crew_pos[0] + dx, crew_pos[1] + dy) for dx, dy in directions if is_valid(crew_pos[0] + dx, crew_pos[1] + dy)]
    prob = 1 / len(possible_moves) if possible_moves else 0
    for move in possible_moves:
        neighbor_index = move[0] * n + move[1]
        P[crew_index, neighbor_index] = prob

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
                actions = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]
                for idx, (dx, dy) in enumerate(actions):
                    new_i, new_j = i + dx, j + dy
                    if 0 <= new_i < n and 0 <= new_j < n and grid[new_i, new_j] != -1:
                        distance = abs(new_i - center[0]) + abs(new_j - center[1])
                        R[current_index, idx] = -(distance - (abs(i - center[0]) + abs(j - center[1])))  # Negative reward for greater distance
    return R, V

def value_iteration(grid,R, P, V, n, gamma=0.95, threshold=0.01):
    num_states = n * n
    P = np.zeros((num_states, num_states))  # Assuming this is predefined elsewhere appropriately
    delta = float('inf')
    
    while delta > threshold:
        delta = 0
        for s in range(num_states):
            v = V[s]
            max_value = float('-inf')
            for a_idx, a in enumerate(get_valid_actions(s, grid, n)):  # Enumerate to get the action index
                expected_value = 0
                next_states = get_next_states(s, a, grid, n)
                for s_prime in next_states:  # Get next states for action a from state s
                    if s_prime < len(P[s]) and P[s, s_prime] > 0:  # Check if s_prime is valid and P[s, s_prime] is positive
                        expected_value += P[s, s_prime] * V[s_prime]  # Sum P(s'|s,a) * V(s')

                action_value = R[s, a_idx] + gamma * expected_value  # R(s,a) + γ * Σ(P(s'|s,a) * V(s'))
                print("R[s, a_idx]:", R[s, a_idx], "Type:", type(R[s, a_idx]))
                print("P[s, s_prime]:", P[s, s_prime], "Type:", type(P[s, s_prime]))
                print("action_value:", action_value, "Type:", type(action_value))
                max_value = max(max_value, action_value)  # Choose the action that maximizes the value
                
            new_v = max_value if max_value != float('-inf') else v
            V[s] = new_v
            delta = max(delta, abs(v - new_v))
    return V

def get_valid_actions(s, grid, n):
    actions = []
    x, y = s // n, s % n
    # Add diagonal movements for the bot if required
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dx, dy in directions:
        if 0 <= x + dx < n and 0 <= y + dy < n and grid[x + dx][y + dy] != -1:
            actions.append((dx, dy))
    return actions

def get_next_states(s, action, grid, n):
    x, y = s // n, s % n
    dx, dy = action
    new_x, new_y = x + dx, y + dy
    if 0 <= new_x < n and 0 <= new_y < n and grid[new_x][new_y] != -1:
        return [new_x * n + new_y]  # Return a list of one next state
    return [s]  # Return the current state in a list if no valid move


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

    return best_move

def simulate_bot_crew_movement(grid, center, bot_toggle = True):
    n = len(grid)
    # Random initial positions for crew and bot, ensuring they are valid
    crew_position, bot_position = initialize_positions(grid, center)
    
    # Initialize reward and value functions
    R, V = initialize_reward_and_value_functions(grid, center)
    
    # Compute the optimal policies using value iteration
    P = initialize_probabilities(grid)  # Make sure this is defined correctly
    V = value_iteration(grid, R, P, V, n)
    
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
            possible_moves = [(dx, dy) for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
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
    
    def is_valid(x, y):
                return 0 <= x < n and 0 <= y < n and grid[x][y] != -1
    
    
    while crew_position != center and steps < 10000:
        new_bot_position = decide_bot_move(grid, bot_position, crew_position, V, n)

        # Determine if the bot is adjacent to the crew
        if abs(crew_position[0] - new_bot_position[0]) + abs(crew_position[1] - new_bot_position[1]) == 1:
            # Calculate the best move to maximize distance from bot, fleeing behavior
            possible_moves = [(dx, dy) for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
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
