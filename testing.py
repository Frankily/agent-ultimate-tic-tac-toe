import argparse
from tictactoe import UltimateTicTacToe
import sys
import mcts
import alphabeta
import dqn

class TestError(Exception):
    pass

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process player names")

    # player one and player two
    parser.add_argument('--player1', type=str, required=True, help="Player One")
    parser.add_argument('--player2', type=str, required=True, help="Player Two")

    # integer argument for the number of tests, defaulting to 10
    parser.add_argument('--count', type=int, default=10, help="Number of times to test")

    # time or depth limit argument for each player
    parser.add_argument('--limit1', type=float, default=1, help="limit parameter")
    parser.add_argument('--limit2', type=float, default=1, help="limit parameter")

    # Parse the arguments
    args = parser.parse_args()
    
    return args

def compare_policies(game, count, p1, p2):
    p1_wins = 0

    for i in range(count):
        p1_policy = p1()
        p2_policy = p2()
        game.reset_game()
        while not game.is_terminal():
            game.display_board()
            if game.current_player == i % 2:
                move = p1_policy(game.get_state())
            else:
                move = p2_policy(game.get_state())
            game.make_move(*move)
        game.display_board()
        if game.payoff() == 0:
            p1_wins += 0.5
        elif (game.payoff() > 0 and i % 2 == 0) or (game.payoff() < 0 and i % 2 == 1):
            p1_wins += 1
    return p1_wins / count

def test_game(game, count, p1_policy_fxn, p2_policy_fxn):
    wins = compare_policies(game, count, p1_policy_fxn, p2_policy_fxn)

    print("WINS: ", wins)

if __name__ == '__main__':
    args = parse_arguments()
    try:
        if args.count < 1:
            raise TestError("count must be positive")
        if args.limit1 < 0:
            raise TestError("time/depth must be positive")
        if args.limit2 < 0:
            raise TestError("time/depth must be positive")
        game = UltimateTicTacToe()
        if args.player1 == 'mcts':
            player1 = mcts.mcts_policy
        elif args.player1 == 'alphabeta':
            player1 = alphabeta.alphabeta_policy
        else:
            player1 = dqn.dqn_policy

        if args.player2 == 'mcts':
            player2 = mcts.mcts_policy
        elif args.player2 == 'alphabeta':
            player2 = alphabeta.alphabeta_policy
        else:
            player2 = dqn.dqn_policy
        
        test_game(game,
                  args.count,
                  lambda: player1(args.limit1),
                  lambda: player2(args.limit2)
                  )
    except TestError as err:
        print(str(err))
        sys.exit(1)