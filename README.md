# agent-ultimate-tic-tac-toe
Research Question:
How does performance of MCTS / minimax with alpha-beta pruning vary with resources for Ultimate Tic Tac Toe? Does DQN work better than these approaches?

The MCTS agent is implemented with the UCT-2 enhancements from lecture and parallelism with multiple trees.
The Minimax with alpha-beta pruning agent is implemented with an optimized heuristic.
The DQN agent is implemented with two DQN agents for each player in an adverserial training. Code was taken from DQN for QFL as template, but model and training script was changed for two agents and with convolutional layers in the neural network.