import math 
import copy
import random
from tictactoe import UltimateTicTacToe

def alphabeta_policy(depth):
    def policy(game):
        return minimax_w_alpha_beta(game, depth, -math.inf, math.inf, True)[1]
    return policy

def minimax_w_alpha_beta(state, depth, alpha, beta, maximizing_player):
    
    if depth == 0 or state.is_terminal():
        
        all_possible_moves = state.get_available_moves()
        move = random.choice(all_possible_moves) if all_possible_moves else None
        
        player = state.current_player
        if not maximizing_player:
            player = 1 - player
        
        opp_player = 1 - player
        
        if state.check_board(state.meta_board, player):
            return math.inf, None
        elif state.check_board(state.meta_board, opp_player):
            return -math.inf, None
        else:
            return evaluate_state(state, player), move
    
    if maximizing_player:
        
        all_possible_moves = state.get_available_moves()
        random.shuffle(all_possible_moves)
        max_score = -math.inf
        best_move = all_possible_moves[0]
        
        for move in all_possible_moves:
            
            new_state = copy.deepcopy(state)
            new_state.make_move(*move)
            
            score, _ = minimax_w_alpha_beta(new_state, depth-1, alpha, beta, False)
            alpha = max(alpha, score)
            
            if score > max_score:
                max_score = score
                best_move = move
            
            if beta <= alpha:
                break
        
        return max_score, best_move
            
    else:
        
        all_possible_moves = state.get_available_moves()
        random.shuffle(all_possible_moves)
        min_score = math.inf
        best_move = all_possible_moves[0]
        
        for move in all_possible_moves:
            
            new_state = copy.deepcopy(state)
            new_state.make_move(*move)
            
            score, _ = minimax_w_alpha_beta(new_state, depth-1, alpha, beta, True)
            beta = min(beta, score)
            
            if score < min_score:
                min_score = score
                best_move = move
            
            if beta <= alpha:
                break
        
        return min_score, best_move
            
    
def evaluate_state(state, player):
    score = 0

    # Evaluate local boards
    for i in range(3):
        for j in range(3):
            board = state.board[i][j]
            for k in range(3):
                # Rows
                if board[k][0] == board[k][1] == player and board[k][2] == -1:
                    score += 10
                if board[k][1] == board[k][2] == player and board[k][0] == -1:
                    score += 10
                if board[k][0] == board[k][2] == player and board[k][1] == -1:
                    score += 10
                # Columns
                if board[0][k] == board[1][k] == player and board[2][k] == -1:
                    score += 10
                if board[1][k] == board[2][k] == player and board[0][k] == -1:
                    score += 10
                if board[0][k] == board[2][k] == player and board[1][k] == -1:
                    score += 10
            # Diagonals
            if board[0][0] == board[1][1] == player and board[2][2] == -1:
                score += 15
            if board[1][1] == board[2][2] == player and board[0][0] == -1:
                score += 15
            if board[0][2] == board[1][1] == player and board[2][0] == -1:
                score += 15
            if board[2][0] == board[1][1] == player and board[0][2] == -1:
                score += 15

    # Evaluate the global meta board
    board = state.meta_board
    
    # Win a square
    for row in board:
        for square in row:
            if square == player:
                score += 40
    
    # Connections of 2
    for k in range(3):
        # Rows
        if board[k][0] == board[k][1] == player and board[k][2] == -1:
            score += 100
        if board[k][1] == board[k][2] == player and board[k][0] == -1:
            score += 100
        if board[k][0] == board[k][2] == player and board[k][1] == -1:
            score += 100
        # Columns
        if board[0][k] == board[1][k] == player and board[2][k] == -1:
            score += 100
        if board[1][k] == board[2][k] == player and board[0][k] == -1:
            score += 100
        if board[0][k] == board[2][k] == player and board[1][k] == -1:
            score += 100
    # Diagonals
    if board[0][0] == board[1][1] == player and board[2][2] == -1:
        score += 150
    if board[1][1] == board[2][2] == player and board[0][0] == -1:
        score += 150
    if board[0][2] == board[1][1] == player and board[2][0] == -1:
        score += 150
    if board[2][0] == board[1][1] == player and board[0][2] == -1:
        score += 150

    return score

if __name__ == "__main__":
    game = UltimateTicTacToe()
    game.display_board()
    player = 1
    
    while not game.is_terminal():
        
        if player == 1:
            minimax_score, minimax_move = minimax_w_alpha_beta(game, 0, -math.inf, math.inf, True)
        else: 
            minimax_score, minimax_move = minimax_w_alpha_beta(game, 5, -math.inf, math.inf, True)
            
        game.make_move(*minimax_move)
        game.display_board()
        player *= -1
        