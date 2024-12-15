# agent-ultimate-tic-tac-toe
## Research Question:
How does performance of MCTS / minimax with alpha-beta pruning vary with resources for Ultimate Tic Tac Toe? How does DQN perform on Ultimate Tic Tac Toe?

## Implementation:
We implemented the MCTS agent with UCT-2 enhancements from lecture and parallelism with multiple trees on separate CPUs, and the implementation can be found in mcts.py.
We implemented the minimax with alpha-beta pruning agent with a heuristic based on two of a kind on the local boards, the winning squares on the meta board, and two of a kind on the meta board. The implementation can be found in alphabeta.py.
We tried to implement two DQN agents with a q-network for each player in an adverserial-style training pipeline. We used the DQN for QFL code as a template, but the model was changed to include convolutional layers on the local and meta board, and the training script was changed to allow training on two agents. The implementation can be found in model.py, dqn.py, and replay.py. 

## Results:
