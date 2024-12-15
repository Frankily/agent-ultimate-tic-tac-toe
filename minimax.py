import random
import math
import copy
import time
from tictactoe import UltimateTicTacToe

terminal_node_count_alpha_beta = 0  # Tracks nodes visited with alpha-beta pruning
terminal_node_count_minimax = 0    # Tracks nodes visited without pruning

def minimax_w_alpha_beta(state, depth, alpha, beta, maximizing_player):
    global terminal_node_count_alpha_beta, terminal_node_count_minimax

    def count_all_terminal_nodes(state, depth):
        """Simulates the standard minimax behavior to count terminal nodes without pruning."""
        global terminal_node_count_minimax
        if depth == 0 or state.is_terminal():
            terminal_node_count_minimax += 1
            return
        for move in state.get_available_moves():
            new_state = copy.deepcopy(state)
            new_state.make_move(*move)
            count_all_terminal_nodes(new_state, depth - 1)

    # Increment the standard minimax counter for the entire subtree
    count_all_terminal_nodes(state, depth)

    # Alpha-beta pruning logic
    if depth == 0 or state.is_terminal():
        terminal_node_count_alpha_beta += 1
        all_possible_moves = state.get_available_moves()
        move = random.choice(all_possible_moves) if all_possible_moves else None

        player = state.current_player
        if not maximizing_player:
            player = 1 - player

        opp_player = 1 - player

        if state.check_board(state.meta_board, player):  # player winning
            return math.inf, None
        elif state.check_board(state.meta_board, opp_player):  # opponent winning
            return -math.inf, None
        elif state.is_terminal():  # draw
            return 0, None
        else:  # non terminal
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

def greedy_heuristic_agent(game, player):

    best_score = -math.inf
    all_possible_moves = game.get_available_moves()
    best_move = random.choice(all_possible_moves)

    for move in all_possible_moves:
        new_state = copy.deepcopy(game)
        new_state.make_move(*move)

        if new_state.check_board(new_state.meta_board, player):
            return math.inf, move    
        else:
            # Evaluate the state for the current player
            current_score = evaluate_state(new_state, player)

        if current_score > best_score:
            best_score = current_score
            best_move = move

    return best_score, best_move


if __name__ == "__main__":
    
    game = UltimateTicTacToe()
    game.display_board()
    player = 1
    p1_wins = 0
    count = 256
    start_time = time.time()
    for i in range(count):
        
        game = UltimateTicTacToe()
        # game.display_board()
        player = 1
        while not game.is_terminal():
            
            if player == 1:
                all_possible_moves = game.get_available_moves()
                minimax_move = random.choice(all_possible_moves)
            else: 
                minimax_score, minimax_move = minimax_w_alpha_beta(game, 4, -math.inf, math.inf, True)
            
            game.make_move(*minimax_move)
            # game.display_board()
            player *= -1
        
        if game.payoff() == 0:
            p1_wins += 0.5
        elif game.payoff() > 0:
            p1_wins += 1
            
        stop_time = time.time()
        
        print(f"Completed {i+1} games ({(i+1)*100 / count:.2f}%). Player 1 wins: {p1_wins * 100/ (i +1) :.2f}. Time elapsed: {stop_time - start_time:.2f}s")
        print(f"Average total terminal nodes searched {terminal_node_count_minimax / (i + 1):.2f}")
        print(f"Average total terminal nodes saved {(terminal_node_count_minimax - terminal_node_count_alpha_beta ) / (i + 1):.2f}")

        