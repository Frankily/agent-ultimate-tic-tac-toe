# agent-ultimate-tic-tac-toe
## Research Question:
How does performance of MCTS / minimax with alpha-beta pruning vary with resources for Ultimate Tic Tac Toe? How does DQN perform on Ultimate Tic Tac Toe?

## Implementation:
We implemented the MCTS agent with UCT-2 enhancements from lecture and parallelism with multiple trees on separate CPUs, and the implementation can be found in mcts.py.
We implemented the minimax with alpha-beta pruning agent with a heuristic based on two in a row on the local boards, the winning squares on the meta board, and two in a row on the meta board. The implementation can be found in alphabeta.py.

We tried to implement two DQN agents with a q-network for each player in an adverserial-style training pipeline. We used the DQN for QFL code as a template, but the model was changed to include convolutional layers on the local and meta board, and the training script was changed to allow training on two agents. The implementation can be found in model.py, dqn.py, and replay.py. 

## Results:

Table 1: Head to head results of a random agent and heuristic-based greedy agent. Results are over 256 games played. 

| Player 1 | Player 2 | Result (percent p1 wins) | Time (s) |
|----------|----------|--------------------------|----------|
| Random   | Greedy   | 0.078  | 2.826    |

The random agent struggles significantly with Ultimate Tic Tac Toe, achieving only a 7.8% win rate against the heuristic-based greedy. For this reason, we used the greedy agent as our baseline agent.

#### Minimax with Alpha-Beta Pruning
Table 2: Head to head results of the greedy agent and the minimax with alpha-beta pruning agent at varying search depth levels. Results are over 256 games played. 
| Player 1 | Player 2   | Depth   | Result (percent p1 wins) | Time (s) |
|----------|------------|---------|--------------------------|----------|
| Greedy   | AlphaBeta  | 2       | 0.18                     | 13       |
| Greedy   | AlphaBeta  | 3       | 0.172                    | 72       |
| Greedy   | AlphaBeta  | 4       | 0.159                    | 224      |
| Greedy   | AlphaBeta  | 5       | 0.133                    | 1107     |
| Greedy   | AlphaBeta  | 6       | 0.118                    | 3857     |

From these results, we observe a steady improvement in win rate as the minimax search depth increase. However, this performance improvement is met with a tradeoff in increasing runtime. 

We also conducted experiments to see how much time, measured in pruned terminal nodes, alpha-beta pruning was saving over standard minimax. We used the random agent so that the opposing player was as unpredictable as possible with respect to which branch it choose to follow. The results are shown in the table below: 

Table 3: Head to head results of the random agent and the minimax agent without alpha-beta pruning at varying search depth levels. Results are over 256 games played.
| Player 1 | Player 2 | Depth | Result (percent p1 wins) | Time (s) | Avg Total Terminal Nodes Searched | Avg Total Terminal Nodes Pruned by Alpha-Beta |
|----------|----------|---------|--------------------------|----------|------------------------------------|---------------------------------------------|
| Random   | Minimax  | 2       | 0.053                    | 206      | 4714.61                            | 4172.86                                     |
| Random   | Minimax  | 3       | 0.052                    | 2208     | 54836.7                            | 51441.9                                     |
| Random   | Minimax  | 4       | 0.050                    | 22816    | 578085                             | 568255.17                                   |


The results were quite staggering. It was almost impractical to run minimax for search depth levels >= 4. And at most times, alpha-beta pruning was pruning > 95% of potential terminal nodes, which almost linearly translates to the time speed-up observed between this table and Table 2.  
