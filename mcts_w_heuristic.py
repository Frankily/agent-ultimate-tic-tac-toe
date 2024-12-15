from math import sqrt, log
import time
import random
import copy
import tictactoe
import threading
from multiprocessing import Pool, cpu_count

# nodes in trees
class Node:
    def __init__(self, state, r=0, n = 0):
        self.state = state
        self.r = r
        self.n = 0
        self.edges = []

# edges in tree
class Edge:
    def __init__(self, action, child):
        self.action = action
        self.child = child
        self.n = 0

# upper confidence bound value for either player - GOOD
def ucb(reward, num_tries, edge_tries, total_tries, player):
    if edge_tries == 0:
        return float('inf')
    exploit = reward / num_tries if num_tries > 0 else 0
    explore = sqrt(2 * log(total_tries) / edge_tries)
    return (exploit if player == 0 else -exploit) + explore

# Function to run MCTS on a single thread
def run_mcts_worker(state, end_time):
    # tree to store unique nodes
    new_state = copy.deepcopy(state)
    state_tree = {}
    root = Node(new_state)
    root_player = new_state[2]
    state_tree[turn_tuple(new_state)] = root
    # expanding tree
    while time.time() < end_time:
        path = [[root],[]]
        leaf = traverse(root, path, end_time)
        if time.time() > end_time: break
        if leaf.n != 0 and not is_terminal(*leaf.state):
            leaf = expand(leaf, state_tree, path, end_time, root_player)
        if time.time() > end_time: break
        reward = simulate(leaf.state, end_time, root_player)
        if time.time() > end_time: break
        update(reward, path)
    
    return root

# mcts policy
def mcts_policy(time_limit, num_processes = 1):
    if num_processes is None:
        num_processes = cpu_count()
    # policy function on state
    def policy(game):
        state = game.get_state()
        end_time = time.time() + time_limit
        with Pool(processes = num_processes) as pool:
            results = pool.starmap(run_mcts_worker, [(state, end_time)] * num_processes)
        # combined stats
        combined_stats = {}
        for root in results: # roots:
            for edge in root.edges:
                action = tuple(edge.action)
                if action not in combined_stats:
                    combined_stats[action] = [0, 0]
                combined_stats[action][0] += edge.child.r
                combined_stats[action][1] += edge.child.n
        # picks the best edge based on current reward calculations
        player = state[2]
        best_action = max(
            combined_stats.keys(),
            key=lambda a: (combined_stats[a][0] / combined_stats[a][1] * (1 - 2 * player))
            if combined_stats[a][1] > 0 else float('-inf')
        )
        return list(best_action)
        
    return policy

# traversing tree by UCB values - GOOD
def traverse(node, path, end_time):
    while node.edges:
        if time.time() > end_time: return None
        total_tries = sum(edge.n for edge in node.edges)
        player = node.state[2]
        best_edge = max(node.edges, key = lambda e: ucb(e.child.r, e.child.n, e.n, total_tries, player))
        node = best_edge.child
        path[1].append(best_edge)
        path[0].append(node)
    return node

# if leaf is expandable - GOOD
def expand(node, tree, path, end_time, root_player):
    if not node.edges:
        for action in get_available_moves(*node.state):
            if time.time() > end_time: return None
            next_state = successor(action, *node.state)
            if turn_tuple(next_state) not in tree:
                tree[turn_tuple(next_state)] = Node(next_state, evaluate_state(*next_state, root_player), 1)
            node.edges.append(Edge(action, tree[turn_tuple(next_state)]))
    rand_edge = random.choice(node.edges)
    path[1].append(rand_edge)
    path[0].append(rand_edge.child)
    return rand_edge.child

# randomly choose actions until terminal state - GOOD
def simulate(state, end_time, root_player):
    while not is_terminal(*state):
        if time.time() > end_time: return None
        action = random.choice(get_available_moves(*state))
        state = successor(action, *state)
    return payoff(*state, root_player)

# update rewards and tries for edges and node on path - GOOD
def update(reward, path):
    for node in path[0]:
        node.r += reward
        node.n += 1
    for edge in path[1]:
        edge.n += 1



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

def payoff(board, meta_board, next_player, last_move, winner, root_player):
    if winner == root_player:
        return 1000
    elif winner == 1 - root_player:
        return -1000
    else:
        return 0
    
def turn_tuple(lst):
    if isinstance(lst, list):
        return tuple(turn_tuple(item) for item in lst)
    return lst

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

def evaluate_state(local_boards, meta_board, player, last_move, winner, root_player):
    score = 0

    if winner == root_player:
        return 1000
    elif winner == 1 - root_player:
        return -1000
    elif is_terminal(local_boards, meta_board, player, last_move, winner):
        return 0
    
    player = root_player
    # Evaluate local boards
    for i in range(3):
        for j in range(3):
            board = local_boards[i][j]
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
    board = meta_board
    
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