import argparse
from tictactoe import UltimateTicTacToe
import sys
import time
import random
import mcts
import alphabeta
import dqn

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process player names")

    # player one and player two
    parser.add_argument('--player1', type=str, required=True, help="Player One")
    parser.add_argument('--player2', type=str, required=True, help="Player Two")

    # integer argument for the number of tests, defaulting to 100
    parser.add_argument('--count', type=int, default=100, help="Number of times to test")

    # time or depth limit argument
    parser.add_argument('--limit', type=int, default=1, help="limit parameter")

    # Parse the arguments
    args = parser.parse_args()
    
    return args

class TestError(Exception):
    pass

if __name__ == '__main__':
    args = parse_arguments()
    try:
        if args.count < 1:
            raise TestError("count must be positive")
        if args.limit < 1:
            raise TestError("time/depth must be positive")
        game = UltimateTicTacToe()
        if args.player1 == 'mcts':
            player1 = mcts.mct_policy
        elif args.player1 == 'alphabeta':
            player1 = alphabeta.alphabeta_policy
        else:
            player1 = dqn.dqn_policy

        if args.player2 == 'mcts':
            player2 = mcts.mct_policy
        elif args.player2 == 'alphabeta':
            player2 = alphabeta.alphabeta_policy
        else:
            player2 = dqn.dqn_policy
        
        test_game(game,
                  args.count,
                  lambda: player1)

    except TestError as err:
        print(str(err))
        sys.exit(1)