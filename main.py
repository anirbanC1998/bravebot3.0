import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import combinations


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
    num_states = n * n
    P = np.zeros((num_states, num_states))

    for i in range(n):
        for j in range(n):
            if grid[i][j] != -1:  # Check if the current cell is open
                current_index = i * n + j
                neighbors = [
                    (i - 1, j), (i + 1, j),  # Up, Down
                    (i, j - 1), (i, j + 1)  # Left, Right
                ]
                valid_neighbors = [
                    (x, y) for x, y in neighbors if 0 <= x < n and 0 <= y < n and grid[x][y] != -1
                ]
                if valid_neighbors:
                    prob = 1 / len(valid_neighbors)
                    for x, y in valid_neighbors:
                        neighbor_index = x * n + y
                        P[current_index, neighbor_index] = prob

    # Normalize probabilities
    for idx in range(n * n):
        if np.sum(P[idx, :]) > 0:  # Avoid division by zero
            P[idx, :] /= np.sum(P[idx, :])
    return P


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


def value_function(grid, center):
    n = len(grid)
    num_states = n * n
    V = np.full((num_states, num_states), 3e2)  # Initial guess is 300 from T_noBot
    # Set known states directly related to the goal (e.g., crew at teleport pad) to zero
    for bot_index in range(num_states):
        teleport_index = center[0] * n + center[1]
        V[bot_index, teleport_index] = 0
    return V


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

                # Evaluate how the bot can influence the crew member towards the center
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_crew_i, new_crew_j = crew_i + di, crew_j + dj
                    if 0 <= new_crew_i < n and 0 <= new_crew_j < n and grid[new_crew_i][new_crew_j] != -1:
                        new_crew_index = new_crew_i * n + new_crew_j
                        distance_to_center = abs(new_crew_i - center_i) + abs(new_crew_j - center_j)

                        # Adjust probability based on whether the bot is adjacent
                        transition_prob = 0.25
                        if is_adjacent:
                            # Increase probability if this direction decreases the distance to the center
                            # Decrease probability otherwise
                            current_distance = abs(crew_i - center_i) + abs(crew_j - center_j)
                            transition_prob *= 2 if distance_to_center < current_distance else 0.9

                        # Calculate the expected time considering reduced distance to center
                        expected_time = transition_prob * (1 + V[bot_index, new_crew_index])
                        possible_times.append(expected_time)

                if possible_times:
                    V[bot_index, crew_index] = min(possible_times)  # Minimize the expected time

        delta = np.max(np.abs(V - V_prev))
        if delta < threshold:
            # print(f"Convergence achieved after {iteration + 1} iterations.")
            break
        # print(f"Iteration {iteration}: Delta = {delta}")

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


def simulate_bot_crew_movement(grid, center, bot_toggle=True):
    n = len(grid)
    # Random initial positions for crew and bot, ensuring they are valid
    crew_position, bot_position = initialize_positions(grid, center)

    # Initialize reward and value functions
    V = value_function(grid, center)

    V = value_iteration(grid, V, n)

    V_reduced = np.full((n, n), np.inf)
    for crew_i in range(n):
        for crew_j in range(n):
            crew_index = crew_i * n + crew_j
            min_time_for_crew = np.min(V[:, crew_index])  # Minimum over all bot positions
            V_reduced[crew_i, crew_j] = min_time_for_crew

    # Using Pandas DataFrame for better print
    df = pd.DataFrame(V_reduced, index=[f"{i}" for i in range(V_reduced.shape[0])],
                      columns=[f"{j}" for j in range(V_reduced.shape[1])])
    print("Expected times to reach the teleport pad for T_Bot:")
    print(df)

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
        # print("Step", steps, "Crew at", crew_position, "Bot at", bot_position)
        if steps % 10 == 0 or crew_position == center:  # Print every 10 steps and at the end
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

    return steps


def main_simulation():
    grid, center = initialize_grid()
    P_no_bot = initialize_probabilities(grid)
    mdv_transition_matrix = solve_for_expected_times(P_no_bot, grid, center)

    # Using Pandas DataFrame for better print
    df = pd.DataFrame(mdv_transition_matrix, index=[f"{i}" for i in range(mdv_transition_matrix.shape[0])],
                      columns=[f"{j}" for j in range(mdv_transition_matrix.shape[1])])
    print("Expected times to reach the teleport pad for T_NoBot:")
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
            min_time_for_crew = np.min(V[:, crew_index])  # Minimum over all bot positions
            V_reduced[crew_i, crew_j] = min_time_for_crew

    # Using Pandas DataFrame for better print
    df = pd.DataFrame(V_reduced, index=[f"{i}" for i in range(V_reduced.shape[0])],
                      columns=[f"{j}" for j in range(V_reduced.shape[1])])
    print("Expected times to reach the teleport pad for T_Bot:")
    print(df)

    # Test each valid position on the grid
    for i in range(n):
        for j in range(n):
            if grid[i][j] != -1:  # Ensure the bot does not start in a blocked cell
                bot_position = (i, j)
                # print(f"Testing bot start position at {bot_position}")
                time_taken = simulate_optimal_bot_crew_movement(grid, bot_position, center, V)
                results[bot_position] = time_taken
                if time_taken < best_time:
                    best_time = time_taken
                    best_position = bot_position

    print(f"Optimal bot position is {best_position} with expected time {best_time}")
    return best_position, results


def training_bot_crew_movement(grid, center, bot_toggle):
    n = len(grid)
    # Random initial positions for crew and bot, ensuring they are valid
    crew_position, bot_position = initialize_positions(grid, center)

    # Print grid for debugging generalized data
    #print_grid(grid, crew_position, bot_position)

    # Initialize reward and value functions
    V = value_function(grid, center)
    V = value_iteration(grid, V, n)

    # Save the Bot and Crew configuration when they succeed
    training_data = []

    steps = 0
    # Have an indicator in training data that indicates a success state
    success = 0
    path = [crew_position]

    # print("Crew starts at:", crew_position)

    def is_valid(x, y):
        return 0 <= x < n and 0 <= y < n and grid[x][y] != -1

    while crew_position != center and steps < 100000:
        if bot_toggle:
            new_bot_position = decide_bot_move(grid, bot_position, crew_position, V, n)
            # Save the action for bot movement
            action = (new_bot_position[0] - bot_position[0], new_bot_position[1] - bot_position[1])
        else:
            new_bot_position = bot_position  # Bot stays put if inactive
            action = (0, 0)

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

        # Record the state and action, success is False, so it will be 0
        training_data.append([bot_position[0], bot_position[1], crew_position[0], crew_position[1], action, 0])

        # Update positions
        bot_position = new_bot_position
        crew_position = new_crew_position
        path.append(crew_position)

        # print_grid(grid, crew_position, bot_position)  # For debugging each step
        steps += 1
        # print("Step", steps, "Crew at", crew_position, "Bot at", bot_position)
        # if steps % 10 == 0 or crew_position == center:  # Print every 10 steps and at the end
        # print("Step", steps, "Crew at", crew_position, "Bot at", bot_position)
        # print_grid(grid, crew_position, bot_position)

        if crew_position == center:
            print("Crew reaches the teleport pad at step, success = True", steps)
            training_data.append([bot_position[0], bot_position[1], crew_position[0], crew_position[1], action, 1])
            # print_grid(grid, crew_position, bot_position)
            break

    if crew_position != center:
        print("Simulation failed")
        # Throw out the training data for failed captures.
        training_data = []

    return training_data


def get_training_data():
    grid, center = initialize_grid()
    # P_no_bot = initialize_probabilities(grid)
    # mdv_transition_matrix = solve_for_expected_times(P_no_bot, grid, center)

    """# Using Pandas DataFrame for better print
    df = pd.DataFrame(mdv_transition_matrix, index=[f"{i}" for i in range(mdv_transition_matrix.shape[0])],
                      columns=[f"{j}" for j in range(mdv_transition_matrix.shape[1])])
    print("Expected times to reach the teleport pad for T_NoBot:")
    print(df)"""

    # Get training data from main simulation into an array
    training_data = training_bot_crew_movement(grid, center, True)

    # Convert data to DataFrame for Neural Network
    df = pd.DataFrame(training_data, columns=['bot_x', 'bot_y', 'crew_x', 'crew_y', 'action', 'success'])

    # Make it into a CSV
    df.to_csv('training_data.csv', index=False)


def generate_ship_configurations(n=11, random_blocks=10, total_configs=10000):
    center = (n // 2, n // 2)  # Center for teleport pad
    fixed_blocks = [(center[0] - 1, center[1] - 1), (center[0] - 1, center[1] + 1),
                    (center[0] + 1, center[1] - 1),
                    (center[0] + 1, center[1] + 1)]  # Fixed blocks around the teleport pad
    training_data = []

    # Always keep open cells around the teleport pad for clear path
    open_paths = [(center[0], center[1] - 1), (center[0], center[1] + 1), (center[0] - 1, center[1]),
                  (center[0] + 1, center[1]),
                  (center[0], center[1] - 2), (center[0], center[1] + 2), (center[0] - 2, center[1]),
                  (center[0] + 2, center[1])]

    # Generate all possible positions except the center and its immediate diagonal surroundings
    all_positions = [(i, j) for i in range(n) for j in range(n)
                     if (i, j) not in fixed_blocks and (i, j) != center and (i, j) not in open_paths]

    # Choose configurations
    chosen_configs = []
    for _ in range(total_configs):
        chosen_blocks = np.random.choice(range(len(all_positions)), random_blocks, replace=False)
        config_blocks = [all_positions[idx] for idx in chosen_blocks] + fixed_blocks
        chosen_configs.append(config_blocks)

    # For each configuration, simulate and collect data
    for config in chosen_configs:
        grid = np.zeros((n, n), dtype=int)
        grid[center] = 0  # Place teleport pad
        for pos in config:
            grid[pos] = -1  # Place blocked cells

        # Assume a function `simulate_bot_crew_movement` that simulates the movement and collects data
        data = training_bot_crew_movement(grid, center, True)
        training_data.extend(data)

    # Convert data to DataFrame for Neural Network
    df = pd.DataFrame(training_data, columns=['bot_x', 'bot_y', 'crew_x', 'crew_y', 'action', 'success'])

    # Make it into a CSV
    df.to_csv('training_data.csv', index=False)


# Classification Bot
class ClassificationBot(nn.Module):
    def __init__(self):
        super(ClassificationBot, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # 4 inputs: Bot Actions and Crew Actions
        self.bn1 = nn.BatchNorm1d(128)  # Batch Normalization
        self.dropout1 = nn.Dropout(0.2)  # Dropout Layer
        self.fc2 = nn.Linear(128, 128) # Duplicate the Layer
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 9)  # 9 outputs: 8 actions + 1 success indicator

    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        return torch.log_softmax(self.fc3(x), dim=1)


def neural_network():
    df = pd.read_csv('training_data.csv')
    # Assuming 'action' is encoded as one-hot and 'success' is a binary column
    X = df[['bot_x', 'bot_y', 'crew_x', 'crew_y']].values
    # Need to merge action and success into a single label set, reduces complexity for the CNN
    y_actions = pd.get_dummies(df['action']).values
    y_success = df['success'].values.reshape(-1, 1) 
    y = np.concatenate((y_actions, y_success), axis=1) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # Initialize model
    model = ClassificationBot()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # Evaluate model
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
    pd.set_option('display.max_rows', None)  # Ensures all rows are displayed
    pd.set_option('display.max_columns', None)  # Ensures all columns are displayed
    pd.set_option('display.width', None)  # Adjusts the display width for wide DataFrames

    np.random.seed(3)  # Fixing Ship Layout, Crew, and Bot Position
    # 0, 3, 5, 12 are good fixed layouts

    # Basic simlulation using fixed Crew and fixed Bot placement
    # main_simulation()

    # Simulate the optimal Bot placement
    # optimal_simulation()

    # Train Neural Network
    #get_training_data()

    # Generalizing Training Data
    generate_ship_configurations()

    neural_network()
