from math import sqrt, log
import time
import random
import copy
import tictactoe
import threading
from concurrent.futures import ThreadPoolExecutor

# nodes in trees
class Node:
    def __init__(self, state):
        self.state = state
        self.r = 0.0
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
def run_mcts_thread(state, end_time, num_threads = 4):
    # tree to store unique nodes
    state_tree = {}
    root = Node(state)
    state_tree[turn_tuple(state)] = root
    # expanding tree
    while time.time() < end_time:
        path = [[root],[]]
        leaf = traverse(root, path, end_time)
        if time.time() > end_time: break
        if leaf.n != 0 and not is_terminal(*leaf.state):
            leaf = expand(leaf, state_tree, path, end_time)
        if time.time() > end_time: break
        reward = simulate(leaf.state, end_time)
        if time.time() > end_time: break
        update(reward, path)
    
    return root

# mcts policy
def mcts_policy(time_limit, num_threads = 4):
    # policy function on state
    def policy(state):
        end_time = time.time() + time_limit
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_roots = [
                executor.submit(run_mcts_thread, state, end_time)
                for _ in range(num_threads)
            ]
        roots = [future.result() for future in future_roots]

        # combined stats
        combined_stats = {}
        for root in roots:
            for edge in root.edges:
                action = tuple(edge.action)  # Convert action list to tuple for dictionary key
                if action not in combined_stats:
                    combined_stats[action] = [0, 0]  # [total_reward, total_visits]
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
def expand(node, tree, path, end_time):
    if not node.edges:
        for action in get_available_moves(*node.state):
            if time.time() > end_time: return None
            next_state = successor(action, *node.state)
            if turn_tuple(next_state) not in tree:
                tree[turn_tuple(next_state)] = Node(next_state)
            node.edges.append(Edge(action, tree[turn_tuple(next_state)]))
    rand_edge = random.choice(node.edges)
    path[1].append(rand_edge)
    path[0].append(rand_edge.child)
    return rand_edge.child

# randomly choose actions until terminal state - GOOD
def simulate(state, end_time):
    while not is_terminal(*state):
        if time.time() > end_time: return None
        action = random.choice(get_available_moves(*state))
        state = successor(action, *state)
    return payoff(*state)

# update rewards and tries for edges and node on path - GOOD
def update(reward, path):
    for node in path[0]:
        node.r += reward
        node.n += 1
    for edge in path[1]:
        edge.n += 1
    
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

def payoff(board, meta_board, next_player, last_move, winner):
    if winner == 0:
        return 1
    elif winner == 1:
        return -1
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