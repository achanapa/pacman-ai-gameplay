import random

SIDE = 8  # Board size
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

class Player:
    def __init__(self, initial_position):
        self.position = initial_position
        self.score = 0
        self.consecutive_coins = 0

    def move(self, dx, dy):
        self.position = (self.position[0] + dx, self.position[1] + dy)

    def add_score(self, points):
        self.consecutive_coins += 1
        self.score += points ** self.consecutive_coins

    def reset_consecutive_coins(self):
        self.consecutive_coins = 0

class State:
    def __init__(self):
        self.players = [Player((0, 0)), Player((SIDE-1, SIDE-1))]
        self.coins = [[random.choice([0, 1]) if random.random() < 0.5 else 2 for _ in range(SIDE)] for _ in range(SIDE)]
        self.coins[0][0] = 0
        self.coins[SIDE-1][SIDE-1] = 0

    def update_coins(self):
        for y in range(SIDE):
            for x in range(SIDE):
                if self.coins[y][x] == 1 and random.random() < 0.5:
                    self.coins[y][x] = 2
                elif self.coins[y][x] == 2 and random.random() < 0.5:
                    self.coins[y][x] = 1

    def copy(self):
        new_state = State()
        new_state.players = [Player(player.position) for player in self.players]
        for i, player in enumerate(self.players):
            new_state.players[i].score = player.score
            new_state.players[i].consecutive_coins = player.consecutive_coins
        new_state.coins = [row[:] for row in self.coins]
        return new_state

    def is_terminal(self):
        return all(coin == 0 for row in self.coins for coin in row)

def is_valid_move(px, py, action, board_size, other_player_position):
    if action == "UP":
        next_position = (px, py + 1)
    elif action == "DOWN":
        next_position = (px, py - 1)
    elif action == "LEFT":
        next_position = (px - 1, py)
    elif action == "RIGHT":
        next_position = (px + 1, py)
    else:
        return False

    if not (0 <= next_position[0] < board_size and 0 <= next_position[1] < board_size):
        return False
    if next_position == other_player_position:
        return False

    return True


def get_valid_moves(px, py, board_size):
    return [action for action in ACTIONS if is_valid_move(px, py, action, board_size)]

def utility(state, player_index):
    opponent_index = 1 - player_index
    return state.players[player_index].score - state.players[opponent_index].score


def state_update(state, player_index, action):
    new_state = state.copy()
    new_state.update_coins()
    player = new_state.players[player_index]
    dx, dy = 0, 0

    if action == "UP":
        dy = 1
    elif action == "DOWN":
        dy = -1
    elif action == "LEFT":
        dx = -1
    elif action == "RIGHT":
        dx = 1

    if is_valid_move(player.position[0], player.position[1], action, SIDE, new_state.players[1 - player_index].position):
        player.move(dx, dy)
        px, py = player.position
        if new_state.coins[py][px] == 1:
            player.add_score(1)
            new_state.coins[py][px] = 0
        else:
            player.reset_consecutive_coins()

    return new_state

class GeneticAlgorithm:
    def __init__(self, state, player_index, population_size=20, genome_length=20, mutation_rate=0.1, generations=30):
        self.state = state
        self.player_index = player_index
        self.population_size = population_size
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        self.generations = generations

    def run(self):
        population = self.initialize_population()
        for _ in range(self.generations):
            fitnesses = [self.evaluate_fitness(genome) for genome in population]
            population = self.select_population(population, fitnesses)
            next_generation = []
            while len(next_generation) < len(population):
                parent1, parent2 = random.sample(population, 2)
                child1, child2 = self.crossover(parent1, parent2)
                next_generation.extend([self.mutate(child1), self.mutate(child2)])
            population = next_generation
        best_genome = max(population, key=lambda g: self.evaluate_fitness(g))
        return best_genome

    def initialize_population(self):
        return [[random.choice(ACTIONS) for _ in range(self.genome_length)] for _ in range(self.population_size)]

    def evaluate_fitness(self, genome):
        simulated_state = self.state.copy()
        player = simulated_state.players[self.player_index]
        score_start = player.score
        for move in genome:
            if not simulated_state.is_terminal():
                simulated_state = state_update(simulated_state, self.player_index, move)
        return player.score - score_start

    def select_population(self, population, fitnesses):
        selected = []
        while len(selected) < len(population):
            tournament = random.sample(list(zip(population, fitnesses)), 3)
            winner = max(tournament, key=lambda item: item[1])[0]
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        point = random.randint(1, self.genome_length - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

    def mutate(self, genome):
        return [gene if random.random() > self.mutation_rate else random.choice(ACTIONS) for gene in genome]


def best_move(state, player_index, depth):
    best_actions = []
    alpha = float('-inf')
    beta = float('inf')
    best_value = float('-inf') if player_index == 0 else float('inf')
    current_player = state.players[player_index]
    other_player = state.players[1 - player_index]
    px, py = current_player.position
    valid_moves = [action for action in ACTIONS if is_valid_move(px, py, action, SIDE, other_player.position)]

    for action in valid_moves:
        new_state = state_update(state, player_index, action)
        move_val = probabilistic_minimax(new_state, player_index, depth - 1, alpha, beta, player_index == 0)
        if (player_index == 0 and move_val > best_value) or (player_index == 1 and move_val < best_value):
            best_value = move_val
            best_actions = [action]
        elif move_val == best_value:
            best_actions.append(action)

        if player_index == 0:
            alpha = max(alpha, move_val)
        else:
            beta = min(beta, move_val)

    return random.choice(best_actions) if best_actions else None

def probabilistic_minimax(state, player_index, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or state.is_terminal():
        return utility(state, player_index)

    if maximizingPlayer:
        expected_value = float('-inf')
        for action in ACTIONS:
            new_state = state_update(state, player_index, action)
            # Calculate expected utility considering the probabilistic nature of coins
            if new_state.coins[new_state.players[player_index].position[1]][new_state.players[player_index].position[0]] == 2:
                # If the coin is transparent, assume a 50% chance it turns solid and is collected next turn
                eval_solid = utility(new_state, player_index) * 0.5 + probabilistic_minimax(new_state, player_index, depth - 1, alpha, beta, False) * 0.5
                eval_transparent = probabilistic_minimax(new_state, player_index, depth - 1, alpha, beta, False)  # Coin remains transparent
                eval = 0.5 * eval_solid + 0.5 * eval_transparent
            else:
                eval = probabilistic_minimax(new_state, player_index, depth - 1, alpha, beta, False)
            
            expected_value = max(expected_value, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return expected_value
    else:
        expected_value = float('inf')
        for action in ACTIONS:
            new_state = state_update(state, 1 - player_index, action)
            if new_state.coins[new_state.players[1 - player_index].position[1]][new_state.players[1 - player_index].position[0]] == 2:
                eval_solid = utility(new_state, player_index) * 0.5 + probabilistic_minimax(new_state, player_index, depth - 1, alpha, beta, True) * 0.5
                eval_transparent = probabilistic_minimax(new_state, player_index, depth - 1, alpha, beta, True)
                eval = 0.5 * eval_solid + 0.5 * eval_transparent
            else:
                eval = probabilistic_minimax(new_state, player_index, depth - 1, alpha, beta, True)
            
            expected_value = min(expected_value, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return expected_value

    
def print_state(state):
    for y in range(SIDE - 1, -1, -1):
        for x in range(SIDE):
            cell = "  "
            if (x, y) == state.players[0].position:
                cell = "P1"
            elif (x, y) == state.players[1].position:
                cell = "P2"
            elif state.coins[y][x] == 1:
                cell = " c"
            elif state.coins[y][x] == 2:
                cell = " _"  # Display transparent coins
            print(f"[{cell}]", end="")
        print()

def game_loop():
    state = State()
    player_turn = 0
    move_count = 0
    depth = 7  # Set the depth for minimax

    while not state.is_terminal():
        if move_count > 70:  # Condition to start using GA
            ga = GeneticAlgorithm(state, player_turn)
            action_sequence = ga.run()
            for action in action_sequence:
                if not state.is_terminal():
                    state = state_update(state, player_turn, action)
                    player_turn = 1 - player_turn
                    move_count += 1
                    print("\nCurrent board (GA active):")
                    print_state(state)
        else:
            action = best_move(state, player_turn, depth)
            state = state_update(state, player_turn, action)
            player_turn = 1 - player_turn
            move_count += 1
            print("\nCurrent board:")
            print_state(state)

    print("\nFinal board:")
    print_state(state)
    print(f"Final scores -> Player 1: {state.players[0].score}, Player 2: {state.players[1].score}")
    winner = "Player 1" if state.players[0].score > state.players[1].score else "Player 2"
    print(f"{winner} wins!")

if __name__ == "__main__":
    game_loop()
