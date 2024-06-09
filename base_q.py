import random
from tqdm import tqdm

class TicTacToe:
    def __init__(self, board_size=3):
        # 0 for empty, 1 for X, -1 for O
        self.board_size = board_size
        self.total_pos = int(board_size ** 2)
        self.reset()

        indices = [i for i in range(0, len(self.board))]
        horizontals = [indices[i:i + self.board_size] for i in range(0, len(indices), self.board_size)]
        verticals = [[i + j * self.board_size for j in range(self.board_size)] for i in range(self.board_size)]
        diagonal_down = [i * self.board_size + i for i in range(self.board_size)]
        diagonal_up = [(i + 1) * (self.board_size - 1) for i in range(self.board_size)]
        lines = horizontals + verticals + [diagonal_down, diagonal_up]
        self.lines = lines


    def is_valid_move(self, position):
        return self.board[position] == 0

    def make_move(self, position):
        if not self.is_valid_move(position):
            return False
        self.board[position] = self.current_player
        self.turns += 1
        return True

    def change_turn(self):
        self.current_player *= -1

    def check_winner(self):
        # Horizontal, vertical, and diagonal checks
        for line in self.lines:
            if all(self.board[i] == self.board[line[0]] and self.board[i] != 0 for i in line):
                self.game_over = self.game_won = True
                return True
        # for a, b, c in self.lines:
        #     if self.board[a] == self.board[b] == self.board[c] and self.board[a] != 0:
        #         return True
        return False

    def get_empty_positions(self):
        empty_pos = [i for i, x in enumerate(self.board) if x == 0]
        if not empty_pos:
            self.game_over = True
        return empty_pos

    def reset(self):
        self.board = [0] * self.total_pos
        self.game_won = False
        self.game_over = False
        self.current_player = 1  # X starts first
        self.turns = 0

    def step(self, action):
        self.make_move(action)
        next_state = self.board.copy()
        next_available_actions = self.get_empty_positions()

        reward = 0
        if self.turns >= self.board_size * 2 - 1:  # The earliest a player can win
            if self.check_winner():
                reward = 1
            elif not next_available_actions: # check for tie
                reward = -0.5
        self.change_turn()

        return next_state, next_available_actions, reward


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        # Separate Q-tables for X and O
        self.q_tables = {1: {}, -1: {}}  # 1 for X, -1 for O
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, player, state, action):
        return self.q_tables[player].get((tuple(state), action), 0)

    def choose_action(self, player, state, available_actions):
        if not available_actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        qs = [self.get_q_value(player, state, a) for a in available_actions]
        max_q = max(qs)
        return available_actions[qs.index(max_q)]


    def update_q_value(self, player, state, action, reward, next_state, next_available_actions, opponent):
        current_q = self.get_q_value(player, state, action)
        best_opponent_action = max([self.get_q_value(opponent, next_state, a) for a in next_available_actions], default=0)
        self.q_tables[player][(tuple(state), action)] = current_q + self.alpha * (reward - self.gamma * best_opponent_action - current_q)


def train_agent(game, agent, episodes=1000):
    for _ in tqdm(range(episodes)):
        game.reset()
        while not game.game_over:
            state_0 = game.board.copy()
            player_0 = game.current_player
            available_actions_0 = game.get_empty_positions()
            action_0 = agent.choose_action(player_0, state_0, available_actions_0)

            if action_0 is not None:
                state_1, available_actions_1, reward_0 = game.step(action_0)

                if game.game_over:
                    agent.update_q_value(player_0, state_0, action_0, reward_0, state_1, available_actions_1, opponent=-player_0)
                    break
                else:
                    player_1 = game.current_player
                    action_1 = agent.choose_action(player_1, state_1, available_actions_1)
                    if action_1 is not None:
                        state_2, available_actions_2, reward_1 = game.step(action_1)

                        # if there's a win we will set reward_0 to -1
                        if game.game_won:
                            reward_0 = -1

                        # do first update
                        agent.update_q_value(player_0, state_0, action_0, reward_0, state_1, available_actions_1, opponent=-player_0)

                        # and second
                        agent.update_q_value(player_1, state_1, action_1, reward_1, state_2, available_actions_2, opponent=-player_1)

            if game.game_over:
                break


def play_against_random(agent, games=100):
    wins = 0
    losses = 0
    ties = 0
    agent.epsilon = 0  # No exploration
    for _ in range(games):
        game = TicTacToe()
        random_player = random.choice([-1, 1])  # Randomly decide who starts first

        while not game.game_over:
            available_actions = game.get_empty_positions()
            if not available_actions:  # No available actions, board is full
                ties += 1
                break

            if game.current_player == random_player:
                # Random player's turn
                action = random.choice(available_actions)
                game.make_move(action)
            else:
                # Agent's turn
                action = agent.choose_action(game.current_player, game.board, available_actions)
                if action is not None:
                    game.make_move(action)
            
            # Check game status
            if game.check_winner():
                if game.current_player != random_player:
                    wins += 1
                elif game.current_player == random_player:
                    losses += 1
                game.game_over = True

            game.change_turn()

    print(f"Out of {games} games:")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Ties: {ties}")
