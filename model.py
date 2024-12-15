import itertools as it
import random as rand

import torch
from torch import nn

class Encoder:
    def __init__(self):
        pass
    def encode(self, state):
        board, meta_board, current_player, last_move, winner = state
        board_encoding = (torch.tensor(board) + 1).view(-1)
        player_enc = torch.tensor(current_player + 1).view(-1)
        last_x = torch.zeros(3)
        last_y = torch.zeros(3)
        if last_move != (None, None):
            last_x[last_move[0]] = 1
            last_y[last_move[1]] = 1
        else:
            for i in range(3):
                last_x[i] = 1
                last_y[i] = 1
        return torch.cat([board_encoding, player_enc, last_x, last_y])

class Model(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self._encoder = encoder
        # Convolutional layer for each local 3x3 board
        self.local_conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        # Convolutional layers for processed local boards
        self.meta_conv = nn.Sequential(
            nn.Conv2d(256, 768, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        total_input_size = 768 + 3 + 3 + 1
        # concatenated layers
        layers = [nn.Linear(total_input_size, 512), nn.ReLU(),
                  nn.Linear(512, 256), nn.ReLU(),
                  nn.Linear(256, 81)]
        self.fc_layers = nn.Sequential(*layers)

        self._opt = torch.optim.Adam(self.parameters(), lr=7.5e-4)
        self._loss_fn = torch.nn.MSELoss()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self._device)

    def forward(self, x):
        board_enc, other_enc = torch.split(x, [81,  x.size(1) - 81], dim=1)
        board = board_enc.view(-1, 1, 9, 9)
        sub_board_features = []
        batch_size = board.size(0)
        for i in range(3):
            for j in range(3):
                sub_board = board[:, :, i*3:(i+1)*3, j*3:(j+1)*3]
                processed_sub_board = self.local_conv(sub_board)
                sub_board_features.append(processed_sub_board)
        sub_board_features = torch.cat(sub_board_features, dim=1)
        sub_board_features = sub_board_features.view(batch_size, 3, 3, -1).permute(0, 3, 1, 2)
        meta_features = self.meta_conv(sub_board_features)
        all_features = torch.cat([meta_features, other_enc], dim=1)
        estimates = self.fc_layers(all_features)
        return estimates
    
    def predict(self, inputs):
        with torch.no_grad():
            encoded_inputs = torch.stack([self._encoder.encode(s) for s in inputs]).to(self._device)
            estimates = self(encoded_inputs)
        return estimates

    def copy(self, m: 'Model'):
        self.load_state_dict(m.state_dict())
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def step(self, batch, target_model):
        # Unpack the batch
        states, actions, next_states, rewards = zip(*batch)

        # Encode inputs and compute current Q-value estimates
        encoded_inputs = torch.stack([self._encoder.encode(s) for s in states]).to(self._device)
        current_q = self(encoded_inputs)
        
        target_est = target_model.predict(next_states)

        expected = []
        masks = []
        for sasr, q_s, curr_est in zip(batch, target_est, current_q):
            q_new = list(curr_est)
            s, a, s2, r = sasr
            action = a[0] * 27 + a[1] * 9 + a[2] * 3 + a[3]
            if is_terminal(*s2):
                q_new[action] = r
            else:
                valid_moves = get_available_moves(*s2)
                max_q = 0
                for big_r, big_c, small_r, small_c in valid_moves:
                    max_q = max(max_q, q_s[big_r * 27 + big_c * 9 + small_r * 3 + small_c])
                q_new[action] = r + max_q
            mask = [0.0] * len(q_s)
            mask[action] = 1.0
            expected.append(q_new)
            masks.append(mask)
        expected = torch.tensor(expected, dtype=torch.float32).to(self._device)
        masks = torch.tensor(masks, dtype=torch.float32).to(self._device)
        masked_loss = torch.sum(((current_q - expected) * masks)**2.0) / len(batch)
#         with open('loss.txt', 'a') as file:
#             file.write(f'{masked_loss}\n')
        # backpropagate
        self._opt.zero_grad()
        masked_loss.backward()
        self._opt.step()


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
