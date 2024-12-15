# agent-ultimate-tic-tac-toe
## Research Question:
How does performance of MCTS / minimax with alpha-beta pruning vary with resources for Ultimate Tic Tac Toe?

## Implementation:
We implemented the MCTS agent with UCT-2 enhancements from lecture and parallelism with multiple trees on separate CPUs, and the implementation can be found in mcts.py.
We implemented the minimax with alpha-beta pruning agent with a heuristic based on two in a row on the local boards, the winning squares on the meta board, and two in a row on the meta board. The implementation can be found in alphabeta.py.

We tried to implement two DQN agents with a q-network for each player in an adverserial-style training pipeline. We used the DQN for QFL code as a template, but the model was changed to include convolutional layers on the local and meta board, and the training script was changed to allow training on two agents. The implementation can be found in model.py, dqn.py, and replay.py. 

## Testing:
For testing, you can run the testing.py script. The player arguments are --player1 and --player2 to determine which agent out of 'mcts', 'alphabeta', and 'mcts_w_h' to run. The argument for number of games is --count. A run of 10 games will not take very long, but 100 games could take around 10 minutes. The constraints for each player are --limit1 and --limit2, which can either be depth or time. The output is the fraction of wins by player 1. It will print out intermediate results too. An example test is shown below:
'python3 testing.py --player1 alphabeta --player2 mcts_w_h --count 10 --limit1 5 --limit2 0.25'

## Results:

#### Table 1: Head to head results of a random agent and heuristic-based greedy agent. Results are over 256 games played. 

| Player 1 | Player 2 | Result (percent p1 wins) | Time (s) |
|----------|----------|--------------------------|----------|
| Random   | Greedy   | 0.078  | 2.826    |

The random agent struggles significantly with Ultimate Tic Tac Toe, achieving only a 7.8% win rate against the heuristic-based greedy. For this reason, we used the greedy agent as our baseline agent.

### Minimax with Alpha-Beta Pruning
#### Table 2: Head to head results of the greedy agent and the minimax with alpha-beta pruning agent at varying search depth levels. Results are over 256 games played. 
| Player 1 | Player 2   | Depth   | Result (percent p1 wins) | Time (s) |
|----------|------------|---------|--------------------------|----------|
| Greedy   | AlphaBeta  | 2       | 18.0%                    | 13       |
| Greedy   | AlphaBeta  | 3       | 17.2%                    | 72       |
| Greedy   | AlphaBeta  | 4       | 15.9%                    | 224      |
| Greedy   | AlphaBeta  | 5       | 13.3%                    | 1107     |
| Greedy   | AlphaBeta  | 6       | 11.8%                    | 3857     |

From these results, we observe a steady improvement in win rate for the minimax with alpha-beta pruning agent as the search depth increase. However, this performance improvement is met with a tradeoff in increasing runtime. 

We also conducted experiments to see how much time, measured in pruned terminal nodes, alpha-beta pruning was saving over standard minimax. We used the random agent so that the opposing player was as unpredictable as possible with respect to which branch it choose to follow. The results are shown in the table below: 

#### Table 3: Head to head results of the random agent and the minimax agent without alpha-beta pruning at varying search depth levels. Results are over 256 games played.
| Player 1 | Player 2 | Depth | Result (percent p1 wins) | Time (s) | Avg Total Terminal Nodes Searched | Avg Total Terminal Nodes Pruned by Alpha-Beta |
|----------|----------|---------|--------------------------|----------|------------------------------------|---------------------------------------------|
| Random   | Minimax  | 2       | 5.3%                    | 206      | 4714.61                            | 4172.86                                     |
| Random   | Minimax  | 3       | 5.2%                    | 2208     | 54836.7                            | 51441.9                                     |
| Random   | Minimax  | 4       | 5.0%                    | 22816    | 578085                             | 568255.17                                   |


The results were quite staggering. It was almost impractical to run minimax for search depth levels >= 4. And at most times, alpha-beta pruning was pruning > 95% of potential terminal nodes, which almost linearly translates to the time speed-up observed between this table and Table 2. We can also observe the exponentially increasing number of terminal nodes searched as the search depth increases. 


### MCTS:
#### Table 4: Head to head results for sensitivity of greedy vs MCTS for time constraints
| Player 1 | Player 2 | Time    | Result (percent p1 wins) | Time (s) |
|----------|----------|---------|--------------------------|----------|
| greedy   | mcts     | 0.125   | 29.67%                   | 678      |
| greedy   | mcts     | 0.25    | 9.81%                    | 1305     |
| greedy   | mcts     | 0.375   | 8.51%                    | 1868     |
Above are the winning percentages over 256 games with 2 CPUs for a range of search times of the MCTS against a greedy player. With the increase in search time for the MCTS, we see a linear increase in time taken to run the games. Also, we see a clear improvement in the agent with more search time from losing almost 30% of the time to losing less than 10% of the time between a search time of 0.125 seconds and a search time of 0.375 seconds.

#### Table 5: Head to head results for Sensitivity of greedy vs MCTS for CPU constraints
| Player 1 | Player 2 | Num CPU | Result (percent p1 wins) | Time (s) |
|----------|----------|---------|--------------------------|----------|
| greedy   | mcts     | 1       | 30.5%                    | 705      |
| greedy   | mcts     | 2       | 29%                      | 678      |
| greedy   | mcts     | 4       | 12.9%                    | 675      |
Above are the winning percentages over 256 games with a standard search time of 0.125 for a range of CPU parallelism of the MCTS against a greedy player. The parallel CPU processes each make separate state trees and run the MCTS algorithm over the several trees. Once the search time is reached, the statistics from the trees are aggregated to determine the best next move. The increase in CPUs does not change the run time of the games, as the calculations are done in parallel. As we include more CPUs, the performance of the MCTS agent improves dramatically too. The small difference in performance betwen 1 and 2 CPUs vs 2 and 4 CPUs could reflect how MCTS may be exploring similar nodes between the trees, but with enough CPUs, the MCTS algorithm will be able to explore broadly most of the nodes in the trees adequately. 

#### Table 6: Head to head results for Sensitivity of greedy vs MCTS for heuristic based traversal
| Player 1 | Player 2 | Time    | Original Result (p1 wins) | Heuristic Result (p1 wins) | Original Time | Heuristic Time |
|----------|----------|---------|---------------------------|----------------------------|---------------|----------------|
| greedy   | mcts     | 0.125   | 29.67%                    | 23.4%                      | 678           | 761            |
| greedy   | mcts     | 0.25    | 9.81%                     | 13.1%                      | 1305          | 1401           |
| greedy   | mcts     | 0.375   | 8.51%                     | 12.3%                      | 1868          | 2089           |




### Comparison of Agents
#### Table 7: Head to head results between the minimax with alpha-beta pruning agent and the MCTS agent. Results are over 2048 games.
| Player 1   | Player 2 | Depth | Time Per Move (s) | Result (percent p1 wins) | Time (s) |
|------------|----------|-------|-------------------|--------------------------|----------|
| AlphaBeta  | MCTS     | 5     | 0.2               | 46.2%                    | 20327    |

Over 2048 games, it appears that the MCTS agent is slightly better than the minimax with alpha-beta pruning agent. 