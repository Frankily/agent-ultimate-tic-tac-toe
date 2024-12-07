class UltimateTicTacToe:
    def __init__(self):
        # Initialize a 3x3 list of 3x3 list to represent the boards.
        self.board = [[[-1 for _ in range(3)] for _ in range(3)] for _ in range(3)]
        self.current_player = 0
        self.previous_turn = None
        self.current_turn = None
        self.winner = None
        self.last_move = (None, None)