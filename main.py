import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Initialize grid for no_bot and bot simulation, with fixed cells and open paths
def initialize_grid(n=11, blocked_cells_count=10):
    grid = np.zeros((n, n))
    center = (n // 2, n // 2)
    grid[center] = 0

    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        grid[center[0] + dx, center[1] + dy] = -1

    flat_indices = np.arange(n * n)
    flat_indices = flat_indices[grid.flatten() != -1]
    blocked_indices = np.random.choice(flat_indices, size=blocked_cells_count, replace=False)
    grid[np.unravel_index(blocked_indices, (n, n))] = -1

    for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
        grid[center[0] + dx, center[1] + dy] = 0

    return grid, center


# Initialize crew and bot positions, bot and crew cannot intersect each other
def initialize_positions(grid, center):
    n = len(grid)
    crew_position = (np.random.randint(n), np.random.randint(n))
    bot_position = (np.random.randint(n), np.random.randint(n))
    while grid[crew_position] == -1 or crew_position == center or crew_position == bot_position:
        crew_position = (np.random.randint(n), np.random.randint(n))
    while grid[bot_position] == -1 or bot_position == center or bot_position == crew_position:
        bot_position = (np.random.randint(n), np.random.randint(n))

    return crew_position, bot_position


# Initialize transition probabilities to solve for expected times and normalize
# In T_NoBot scenario
def initialize_probabilities(grid):
    n = len(grid)
    num_states = n * n
    P = np.zeros((num_states, num_states))

    for i in range(n):
        for j in range(n):
            if grid[i][j] != -1:
                current_index = i * n + j
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < n and 0 <= y < n and grid[x][y] != -1]
                if valid_neighbors:
                    prob = 1 / len(valid_neighbors)
                    for x, y in valid_neighbors:
                        neighbor_index = x * n + y
                        P[current_index, neighbor_index] = prob

    for x in range(num_states):
        if np.sum(P[x, :]) > 0:
            P[x, :] /= np.sum(P[x, :])
    return P


# Solve for expected times using P, zero out the center and relative to the amount of cells in the grid,
# calculate the expected times using matrix algebra
def solve_for_expected_times(P, grid, center):
    n = len(grid)
    num_states = n * n
    A = np.zeros((num_states, num_states))
    b = np.ones(num_states)

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            if grid[i, j] == -1: 
                A[idx, idx] = 1
                b[idx] = 0
            elif (i, j) == center:
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = 0
            else:
                A[idx, idx] = 1
                for (x, y) in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if 0 <= x < n and 0 <= y < n and grid[x, y] != -1:
                        neighbor_idx = x * n + y
                        A[idx, neighbor_idx] = -P[idx, neighbor_idx]

    T = np.linalg.solve(A, b)
    return T.reshape((n, n))

# Set the initial guess for values at 300, relative to the high possible expected steps from T_noBot
# and set the center to 0 as always
def value_function(grid, center):
    n = len(grid)
    num_states = n * n
    V = np.full((num_states, num_states), 3e2)
    for bot_index in range(num_states):
        teleport_index = center[0] * n + center[1]
        V[bot_index, teleport_index] = 0
    return V

# Perform value_iterations for each T_bot_crew(cell) by prioritizing shortening the distance
# between the crew member and center, and sticking to the crew member when adjacent
# Makes sure to only input the minimized possible time in each cell

def value_iteration(grid, V, n, threshold=0.001, max_iterations=300):
    num_states = n * n
    center_i, center_j = n // 2, n // 2

    for iteration in range(max_iterations):
        V_prev = np.copy(V)

        for bot_index in range(num_states):
            bot_i, bot_j = bot_index // n, bot_index % n
            if grid[bot_i][bot_j] == -1:
                continue

            for crew_index in range(num_states):
                crew_i, crew_j = crew_index // n, crew_index % n
                if grid[crew_i][crew_j] == -1:
                    continue

                is_adjacent = False
                if abs(bot_i - crew_i) + abs(bot_j - crew_j) == 1: is_adjacent = True
                possible_times = []

                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_crew_i, new_crew_j = crew_i + di, crew_j + dj
                    if 0 <= new_crew_i < n and 0 <= new_crew_j < n and grid[new_crew_i][new_crew_j] != -1:
                        new_crew_index = new_crew_i * n + new_crew_j
                        distance_to_center = abs(new_crew_i - center_i) + abs(new_crew_j - center_j)

                        # Adjust probability based on whether the bot is adjacent
                        transition_prob = 0.25
                        if is_adjacent:
                            current_distance = abs(crew_i - center_i) + abs(crew_j - center_j)
                            transition_prob *= 2 if distance_to_center < current_distance else 0.9
                        expected_time = transition_prob * (1 + V[bot_index, new_crew_index])
                        possible_times.append(expected_time)

                if possible_times:
                    V[bot_index, crew_index] = min(possible_times)

        delta = np.max(np.abs(V - V_prev))
        if delta < threshold:
            print(f"Convergence achieved after {iteration + 1} iterations.")
            break
        # print(f"Iteration {iteration}: Delta = {delta}")

    return V

# Print the grid at each time step

def print_grid(grid, crew_position, bot_position):
    display_grid = np.array([' ' if x == 0 else 'X' for x in grid.flatten()]).reshape(grid.shape)
    display_grid[crew_position] = 'C'
    display_grid[bot_position] = 'B'  
    center = (len(grid) // 2, len(grid) // 2)
    display_grid[center] = 'T'
    df = pd.DataFrame(display_grid, index=[f"{i}" for i in range(grid.shape[0])],
                      columns=[f"{j}" for j in range(grid.shape[1])])
    print(df)
    print()


# Bot movement logic, decided by picking the min value from the value matrix

def decide_bot_move(grid, bot_position, crew_position, V, n):
    min_time = float('inf')
    best_move = bot_position

    bot_x, bot_y = bot_position
    crew_x, crew_y = crew_position
    crew_index = crew_x * n + crew_y

    for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
        new_bot_x, new_bot_y = bot_x + dx, bot_y + dy
        if 0 <= new_bot_x < n and 0 <= new_bot_y < n and grid[new_bot_x][new_bot_y] != -1:
            new_bot_index = new_bot_x * n + new_bot_y
            if V[new_bot_index, crew_index] < min_time:
                min_time = V[new_bot_index, crew_index]
                best_move = (new_bot_x, new_bot_y)

    return best_move


# Main simulation that has a toggle for the Bot to exist or not
# Has a minimum of 100000 timesteps to debug bad grids.
# Implements the crew fleeing logic according to position of bot

def simulate_bot_crew_movement(grid, center, bot_toggle=True):
    n = len(grid)
    crew_position, bot_position = initialize_positions(grid, center)
    V = value_function(grid, center)
    V = value_iteration(grid, V, n)

    V_reduced = np.full((n, n), np.inf)
    for crew_i in range(n):
        for crew_j in range(n):
            crew_index = crew_i * n + crew_j
            min_time_for_crew = np.min(V[:, crew_index])
            V_reduced[crew_i, crew_j] = min_time_for_crew

    df = pd.DataFrame(V_reduced, index=[f"{i}" for i in range(V_reduced.shape[0])],
                      columns=[f"{j}" for j in range(V_reduced.shape[1])])
    print("Expected times to reach the teleport pad for T_Bot:")
    print(df)

    steps = 0
    path = [crew_position]
    bot_path = [bot_position]
    print("Crew starts at:", crew_position)
    print("Bot starts at:", bot_position)
    print_grid(grid, crew_position, bot_position)

    def is_valid(x, y):
        return 0 <= x < n and 0 <= y < n and grid[x][y] != -1

    while crew_position != center and steps < 100000:
        if bot_toggle:
            new_bot_position = decide_bot_move(grid, bot_position, crew_position, V, n)
        else:
            new_bot_position = bot_position  

        
        if abs(crew_position[0] - new_bot_position[0]) + abs(crew_position[1] - new_bot_position[1]) == 1:
            
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
           
            possible_moves = [(dx, dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                              if is_valid(crew_position[0] + dx, crew_position[1] + dy)]
            move_choice = possible_moves[np.random.choice(len(possible_moves))] if possible_moves else (0, 0)

        new_crew_position = (crew_position[0] + move_choice[0], crew_position[1] + move_choice[1])

        
        bot_position = new_bot_position
        crew_position = new_crew_position
        bot_path.append(bot_position)
        path.append(crew_position)

        # print_grid(grid, crew_position, bot_position)  
        steps += 1
        # print("Step", steps, "Crew at", crew_position, "Bot at", bot_position)
        if steps % 10 == 0 or crew_position == center:
            print("Step", steps, "Crew at", crew_position, "Bot at", bot_position)
            # print_grid(grid, crew_position, bot_position)

        if crew_position == center:
            print("Crew reaches the teleport pad at step", steps)
            print_grid(grid, crew_position, bot_position)
            break

    return path, steps


# Intialize crew only for optimal_simulation Bot simulation

def initialize_crew_position(grid, center):
    n = len(grid)
    crew_position = (np.random.randint(n), np.random.randint(n))

    while grid[crew_position] == -1 or crew_position == center:
        crew_position = (np.random.randint(n), np.random.randint(n))

    return crew_position

# Main simulation used for optimal_simulation, returns steps taken only to compare
# to every other 'steps' for each vaild cell the bot can spawn in
# Contains crew fleeing and random movement logic

def simulate_optimal_bot_crew_movement(grid, bot_position, center, V):
    
    n = len(grid)
    crew_position = initialize_crew_position(grid, center)
    #print(crew_position)

    steps = 0

    def is_valid(x, y):
        return 0 <= x < n and 0 <= y < n and grid[x][y] != -1

    while crew_position != center and steps < 10000:
        new_bot_position = decide_bot_move(grid, bot_position, crew_position, V, n)

        
        if abs(crew_position[0] - new_bot_position[0]) + abs(crew_position[1] - new_bot_position[1]) == 1:
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
            
            possible_moves = [(dx, dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                              if is_valid(crew_position[0] + dx, crew_position[1] + dy)]
            move_choice = possible_moves[np.random.choice(len(possible_moves))] if possible_moves else (0, 0)

        new_crew_position = (crew_position[0] + move_choice[0], crew_position[1] + move_choice[1])

        bot_position = new_bot_position
        crew_position = new_crew_position
        steps += 1

    return steps

# Main simulation to test optimal path for the Bot

def main_simulation():
    grid, center = initialize_grid()
    P_no_bot = initialize_probabilities(grid)
    mdv_transition_matrix = solve_for_expected_times(P_no_bot, grid, center)

    
    df = pd.DataFrame(mdv_transition_matrix, index=[f"{i}" for i in range(mdv_transition_matrix.shape[0])],
                      columns=[f"{j}" for j in range(mdv_transition_matrix.shape[1])])
    print("Expected times to reach the teleport pad for T_NoBot:")
    print(df)

    # Turn the Bot On or Off
    bot_toggle = True
    path, steps = simulate_bot_crew_movement(grid, center, bot_toggle)
    print(f"Path : {path}, Steps: {steps}")

# Main simulation to test the best bot spawn in open grids given the same random crew spawn
# Returns the best steps taken and bot spawn position

def optimal_simulation():
    
    grid, center = initialize_grid()
    P_no_bot = initialize_probabilities(grid)
    mdv_transition_matrix = solve_for_expected_times(P_no_bot, grid, center)

    df = pd.DataFrame(mdv_transition_matrix, index=[f"{i}" for i in range(mdv_transition_matrix.shape[0])],
                      columns=[f"{j}" for j in range(mdv_transition_matrix.shape[1])])
    print("Expected times to reach the teleport pad for T_NoBot:")
    print(df)

    n = len(grid)
    best_position = None
    best_time = float('inf')
    results = {}

    V = value_function(grid, center)
    V = value_iteration(grid, V, n)

    V_reduced = np.full((n, n), np.inf)
    for crew_i in range(n):
        for crew_j in range(n):
            crew_index = crew_i * n + crew_j
            min_time_for_crew = np.min(V[:, crew_index])
            V_reduced[crew_i, crew_j] = min_time_for_crew

    
    df = pd.DataFrame(V_reduced, index=[f"{i}" for i in range(V_reduced.shape[0])],
                      columns=[f"{j}" for j in range(V_reduced.shape[1])])
    print("Expected times to reach the teleport pad for T_Bot:")
    print(df)

    
    for i in range(n):
        for j in range(n):
            if grid[i][j] != -1:
                bot_position = (i, j)
                # print(f"Testing bot start position at {bot_position}")
                time_taken = simulate_optimal_bot_crew_movement(grid, bot_position, center, V)
                results[bot_position] = time_taken
                if time_taken < best_time:
                    best_time = time_taken
                    best_position = bot_position

    print(f"Optimal bot position is {best_position} with expected time {best_time}")
    return best_position, results

# Training simulation for recording data for Learned Bot and Generalizing.. Bot

def training_bot_crew_movement(grid, center, bot_toggle):
    n = len(grid)
    crew_position, bot_position = initialize_positions(grid, center)

    # print_grid(grid, crew_position, bot_position)

    
    V = value_function(grid, center)
    V = value_iteration(grid, V, n)

    
    training_data = []
    steps = 0
    path = [crew_position]

    # print("Crew starts at:", crew_position)

    def is_valid(x, y):
        return 0 <= x < n and 0 <= y < n and grid[x][y] != -1

    while crew_position != center and steps < 100000:
        if bot_toggle:
            new_bot_position = decide_bot_move(grid, bot_position, crew_position, V, n)
            action = (new_bot_position[0] - bot_position[0], new_bot_position[1] - bot_position[1])
        else:
            new_bot_position = bot_position
            action = (0, 0)

        if abs(crew_position[0] - new_bot_position[0]) + abs(crew_position[1] - new_bot_position[1]) == 1:
            
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
            
            possible_moves = [(dx, dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                              if is_valid(crew_position[0] + dx, crew_position[1] + dy)]
            move_choice = possible_moves[np.random.choice(len(possible_moves))] if possible_moves else (0, 0)

        new_crew_position = (crew_position[0] + move_choice[0], crew_position[1] + move_choice[1])

        # Record the state and action, 0 represents success state, so 0 because crew is not at the center
        training_data.append([bot_position[0], bot_position[1], crew_position[0], crew_position[1], action, 0])

        bot_position = new_bot_position
        crew_position = new_crew_position
        path.append(crew_position)

        # print_grid(grid, crew_position, bot_position)  # For debugging each step
        steps += 1
        # print("Step", steps, "Crew at", crew_position, "Bot at", bot_position)
        # if steps % 10 == 0 or crew_position == center:
        # print("Step", steps, "Crew at", crew_position, "Bot at", bot_position)
        # print_grid(grid, crew_position, bot_position)

        if crew_position == center:
            print("Crew reaches the teleport pad at step, success = True", steps)
            training_data.append([bot_position[0], bot_position[1], crew_position[0], crew_position[1], action, 1]) # 1 equals success
            # print_grid(grid, crew_position, bot_position)
            break

    if crew_position != center:
        print("Simulation failed")
        # Throw out the training data for failed captures.
        training_data = []

    return training_data

# Get training data for Learned Bot

def get_training_data():
    grid, center = initialize_grid()

    training_data = training_bot_crew_movement(grid, center, True)

    df = pd.DataFrame(training_data, columns=['bot_x', 'bot_y', 'crew_x', 'crew_y', 'action', 'success'])

    df.to_csv('training_data.csv', index=False)

# Generalize valid ship configurations and run simulations on them. Throw up simulation data that ends in failure or takes > 100000 steps

def generate_ship_configurations(n=11, random_blocks=10, total_configs=10000):
    center = (n // 2, n // 2) 
    fixed_blocks = [(center[0] - 1, center[1] - 1), (center[0] - 1, center[1] + 1), (center[0] + 1, center[1] - 1), (center[0] + 1, center[1] + 1)]  
    training_data = []

    open_paths = [(center[0], center[1] - 1), (center[0], center[1] + 1), (center[0] - 1, center[1]), (center[0] + 1, center[1]),
                  (center[0], center[1] - 2), (center[0], center[1] + 2), (center[0] - 2, center[1]), (center[0] + 2, center[1])]

    all_positions = [(i, j) for i in range(n) for j in range(n)
                     if (i, j) not in fixed_blocks and (i, j) != center and (i, j) not in open_paths]
    
    chosen_configs = []
    for _ in range(total_configs):
        chosen_blocks = np.random.choice(range(len(all_positions)), random_blocks, replace=False)
        config_blocks = [all_positions[idx] for idx in chosen_blocks] + fixed_blocks
        chosen_configs.append(config_blocks)
    
    for config in chosen_configs:
        grid = np.zeros((n, n), dtype=int)
        grid[center] = 0  
        for pos in config:
            grid[pos] = -1  

        data = training_bot_crew_movement(grid, center, True)
        training_data.extend(data)
   
    df = pd.DataFrame(training_data, columns=['bot_x', 'bot_y', 'crew_x', 'crew_y', 'action', 'success'])

    df.to_csv('training_data.csv', index=False)


# Classification Bot, utilizes Linear, Batch norm, DO, Relu
class ClassificationBot(nn.Module):
    def __init__(self):
        super(ClassificationBot, self).__init__()
        self.fc1 = nn.Linear(4, 128)  
        self.bn1 = nn.BatchNorm1d(128)  
        self.dropout1 = nn.Dropout(0.2)  
        self.fc2 = nn.Linear(128, 128)  
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 9)  

    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        return torch.log_softmax(self.fc3(x), dim=1)

# Neural network, outputs accuracy over data provided, only match binary values as output to simplify complexity
# Success parameter is to include only successful data in training data, more for generalizing scenario
# Normalizes training data before training

def neural_network():
    df = pd.read_csv('training_data.csv')
    X = df[['bot_x', 'bot_y', 'crew_x', 'crew_y']].values
    y_actions = pd.get_dummies(df['action']).values
    y_success = df['success'].values.reshape(-1, 1)
    y = np.concatenate((y_actions, y_success), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    model = ClassificationBot()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    model.eval()
    results = []
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_test.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        print(f"Test Accuracy: {accuracy}")
        results.extend(zip(X_test.numpy(), predicted.numpy(), labels.numpy()))

    results_df = pd.DataFrame(results, columns=['Inputs', 'Predicted', 'Actual'])
    results_df.to_csv('model_output.csv', index=False)


if __name__ == "__main__":
    pd.set_option('display.max_rows', None) 
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.width', None)

    np.random.seed(3)  # Fixing Ship Layout, Crew, and Bot Position
    # 0, 3, 5, 12 are good fixed layouts

    # Uncomment what you want to run
    # Bear in mind, generate_ship... takes 3 hours.
    # That means neural_network() would take around 2-3 hours more.
    main_simulation()
    optimal_simulation()
    get_training_data()
    #generate_ship_configurations()
    neural_network()
