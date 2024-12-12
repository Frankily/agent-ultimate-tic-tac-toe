import random
import copy
from model_2 import Model, Encoder
from replay import ReplayDB
import tictactoe
import time

# SOME TIC TAC TOE FUNCTIONS THAT ARE BETTER IMPLEMENTED HERE
def is_terminal(board, meta_board, next_player, last_move, winner):
    if winner != -1:
        return True
    if last_move != (None, None):
        for small_row in range(3):
            for small_col in range(3):
                if board[last_move[0]][last_move[1]][small_row][small_col] == -1:
                    return False
    else:
        for large_row in range(3):
            for large_col in range(3):
                if meta_board[large_row][large_col] == -1:
                    for small_row in range(3):
                        for small_col in range(3):
                            if board[large_row][large_col][small_row][small_col] == -1:
                                return False
    return True

def check_board(board, player):
    # check rows
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] == player:
            return True
    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] == player:
            return True
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] == player:
        return True
    if board[0][2] == board[1][1] == board[2][0] == player:
        return True
    return False

def successor(move, board, meta_board, current_player, last_move, winner):
    new_board = copy.deepcopy(board)
    new_meta_board = copy.deepcopy(meta_board)
    last_move = (move[2], move[3])
    next_winner = winner
    next_player = 1 - current_player
    new_board[move[0]][move[1]][move[2]][move[3]] = current_player
    if check_board(new_board[move[0]][move[1]], current_player):
        new_meta_board[move[0]][move[1]] = current_player
        if check_board(new_meta_board, current_player):
            next_winner = current_player
    if new_meta_board[move[2]][move[3]] != -1:
        last_move = (None, None)
    return [new_board, new_meta_board, next_player, last_move, next_winner]

def get_available_moves(board, meta_board, next_player, last_move, winner):
    available_moves = []
    if last_move != (None, None):
        for small_row in range(3):
            for small_col in range(3):
                if board[last_move[0]][last_move[1]][small_row][small_col] == -1:
                    available_moves.append([last_move[0], last_move[1], small_row, small_col])
    else:
        for large_row in range(3):
            for large_col in range(3):
                if meta_board[large_row][large_col] == -1:
                    for small_row in range(3):
                        for small_col in range(3):
                            if board[large_row][large_col][small_row][small_col] == -1:
                                available_moves.append([large_row, large_col, small_row, small_col])
    return available_moves

def payoff(board, meta_board, next_player, last_move, winner):
    if winner == 0:
        return 1
    elif winner == 1:
        return -1
    else:
        return 0

def epsilon_greedy(eps):
    def select(values):
        if (random.random() < eps):
            return random.choice(values)[0]
        else:
            return max(values, key=lambda x: x[1])[0]
    return select

class DQN:
    def __init__(self, encoder):
        self._encoder = encoder
        self._game = tictactoe.UltimateTicTacToe()
        self._game_initial = self._game.get_state()

        self._target_0 = Model(encoder)
        self._target_0.eval()
        self._learning_0 = Model(encoder)
        self._learning_0.train()

        self._target_1 = Model(encoder)
        self._target_1.eval()

        self._learning_1 = Model(encoder)
        self._learning_1.train()
        self._loaded = 0
    
    def get_second(self):
        select = epsilon_greedy(0.1)
        initial_state = self._game_initial
        values = self._learning_0.predict([initial_state])[0]
        moves = get_available_moves(*initial_state)
        move_values = [[move, value] for move, value in zip(moves, values)]
        action = select(move_values)
        return successor(action, *initial_state)

    def train(self, replay_0, replay_1, games_at_once = 4, epsilon = 0.5, train_interval = 4, train_batch = 32, transfer_interval = 8192, episodes = 40960):
        term_count = 0
        action_count = 0
        next_train = train_interval
        next_xfer = transfer_interval
        select = epsilon_greedy(epsilon)
        previous_states = [] # 1 
        previous_actions = [] # 1
        previous_rewards = [] # 1
        current_states = [self._game.get_state() for i in range(games_at_once)] # 0
        start_time = time.time()
        while term_count < episodes:
            # person 0 takes an action
            values = self._learning_0.predict(current_states)
            moves = [get_available_moves(*state) for state in current_states]
            move_values = []
            for i in range(len(current_states)):
                move_value = []
                for move in moves[i]:
                    move_value.append([move, values[i][27 * move[0] + 9 * move[1] + 3 * move[2] + move[3]]])
                move_values.append(move_value)
            current_actions = [select(move_value) for move_value in move_values] # 0
            new_states = [successor(action, *state) for action, state in zip(current_actions, current_states)] # 1
            terminal_0 = [is_terminal(*state) for state in new_states] # 0
            current_rewards = [(payoff(*s) if t else 0) for s, t in zip(new_states, terminal_0)] # 0
            for s, a, s2, r, t in zip(current_states, current_actions, new_states, current_rewards, terminal_0):
                if t:
                    replay_0.add(s, a, s2, r)
            # person 1 updates 
            if previous_states != []:
                for s, a, s1, s2, r, r1 in zip(previous_states, previous_actions, current_states, new_states, previous_rewards, current_rewards):
                    if not is_terminal(*s):
                        if is_terminal(*s1):
                            replay_1.add(s, a, s1, -r)
                        else:
                            replay_1.add(s, a, s2, -r1)
            
            # person 1 acts
            new_states_2 = [self.get_second() if t else s for s, t in zip(new_states, terminal_0)]
            values = self._learning_1.predict(new_states_2)
            moves = [get_available_moves(*state) for state in new_states_2]
            move_values = []
            for i in range(len(new_states_2)):
                move_value = []
                for move in moves[i]:
                    move_value.append([move, values[i][27 * move[0] + 9 * move[1] + 3 * move[2] + move[3]]])
                move_values.append(move_value)
            update_actions = [select(move_value) for move_value in move_values]
            update_states = [successor(action, *state) for action, state in zip(update_actions, new_states_2)]
            terminal_1 = [is_terminal(*state) for state in update_states]
            update_rewards = [(payoff(*s) if t else 0) for s, t in zip(update_states, terminal_1)] # 1
            for s, a, s2, r, t in zip(current_states, current_actions, update_states, update_rewards, terminal_0):
                if not t:
                    replay_0.add(s, a, s2, r)
            # switch back
            previous_states = new_states
            previous_actions = update_actions
            previous_rewards = update_rewards
            current_states = [self._game_initial if t else s for s, t in zip(update_states, terminal_1)]
            term_count += sum(1 if curr_t else 0 for curr_t in terminal_0)
            action_count += games_at_once

            if action_count >= next_train and action_count > train_batch:
                samples_0 = replay_0.sample(train_batch)
                self._learning_0.step(samples_0, self._target_0)
                next_train += train_interval
                samples_1 = replay_1.sample(train_batch)
                self._learning_1.step(samples_1, self._target_1)             
                epsilon = max(epsilon * 0.999, 0.1)
                select = epsilon_greedy(epsilon)

            if action_count >= next_xfer:
                self._target_0.copy(self._learning_0)
                self._target_1.copy(self._learning_1)
                next_xfer += transfer_interval
            
            if term_count % 4 == 0:
                end_time = time.time()
                print(end_time - start_time, term_count)

    def save_models(self, learning_0_path, learning_1_path):
        self._learning_0.save(learning_0_path)
        self._learning_1.save(learning_1_path)

    def load_models(self, learning_0_path, learning_1_path):
        self._learning_0.load(learning_0_path)
        self._learning_1.load(learning_1_path)
        self._target_0.copy(self._learning_0)
        self._target_1.copy(self._learning_1)
        self._loaded = 1

    def dqn_policy(self, player):
        if self._loaded == 0:
            self.load_models('player_0_1.pth', 'player_1_1.pth')
        if player == 0:
            model = self._learning_0
        else:
            model = self._learning_1
        def choose_action(s):
            select = epsilon_greedy(0)
            values = model.predict([s])[0]
            moves = get_available_moves(*s)
            move_values = []
            for move in moves:
                move_values.append([move, values[27 * move[0] + 9 * move[1] + 3 * move[2] + move[3]]])
            return select(move_values)
        return choose_action


if __name__ == "__main__":
    encoder = Encoder()
    replay_0 = ReplayDB(100000)
    replay_1 = ReplayDB(100000)
    dqn = DQN(encoder)
    dqn.train(replay_0, replay_1)
    dqn.save_models('play_0.pth', 'play_1.pth')