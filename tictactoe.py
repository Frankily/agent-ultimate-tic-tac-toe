class UltimateTicTacToe:
    # Initialize a 3x3 list of 3x3 list to represent the boards.
    def __init__(self):
        self.board = [[[[-1 for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
        self.meta_board = [[-1 for _ in range(3)] for _ in range(3)]
        self.current_player = 0 # 0 for player 'O', 1 for player 'X'
        self.previous_turn = None
        self.winner = -1
        self.last_move = (None, None)
    
    def get_state(self):
        return self.board, self.meta_board, self.current_player, self.winner, self.last_move

    # display ultimate tic tac toe board
    def display_board(self):
        def num_to_symbol(num):
            if num == -1:
                return ' '
            elif num == 0:
                return 'O'
            elif num == 1:
                return 'X'

        def get_large_symbol(player, row):
            if player == 0:  # O
                large_O = ["    OOO    ", "   O   O   ", "    OOO    "]
                return large_O[row]
            elif player == 1:  # X
                large_X = ["   X   X   ", "     X     ", "   X   X   "]
                return large_X[row]
            return None

        for i in range(3):  # row of big board
            for y in range(3):  # row within each small board
                for j in range(3):  # column of big board
                    if self.meta_board[i][j] != -1:
                        symbol_row = get_large_symbol(self.meta_board[i][j], y)
                        print(f"{symbol_row}", end="")
                    else:
                        for x in range(3):  # column within each small board
                            symbol = num_to_symbol(self.board[i][j][y][x])
                            if x < 2:
                                print(f" {symbol} |", end="")
                            else:
                                print(f" {symbol} ", end="")
                    if j < 2:
                        print("||", end="")
                print()
            if i < 2:
                print("-" * 11 + "||" + "-" * 11 + "||" + "-" * 11)
        print()
    
    # check if a board is won
    def check_board(self, board, player):
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

    # get available moves
    def get_available_moves(self):
        available_moves = []
        if self.last_move != (None, None):
            for small_row in range(3):
                for small_col in range(3):
                    if self.board[self.last_move[0]][self.last_move[1]][small_row][small_col] == -1:
                        available_moves.append([self.last_move[0], self.last_move[1], small_row, small_col])
        else:
            for large_row in range(3):
                for large_col in range(3):
                    if self.meta_board[large_row][large_col] == -1:
                        for small_row in range(3):
                            for small_col in range(3):
                                if self.board[large_row][large_col][small_row][small_col] == -1:
                                    available_moves.append([large_row, large_col, small_row, small_col])
        return available_moves

    # Make a move for the current player.
    def make_move(self, big_row, big_col, small_row, small_col):
        if self.meta_board[big_row][big_col] != -1:
            raise ValueError("Mini board is already taken!")
        if self.board[big_row][big_col][small_row][small_col] != -1:
            raise ValueError("Cell is already taken!")
        if self.last_move != (None, None) and (big_row, big_col) != self.last_move:
            raise ValueError("Did not play in write mini board!")

        # Place the current player's symbol in the specified cell.
        self.board[big_row][big_col][small_row][small_col] = self.current_player

        # Change turn
        self.previous_turn = (big_row, big_col, small_row, small_col)
        self.last_move = (small_row, small_col)  # Determine where the next move can be made
        if self.meta_board[small_row][small_col] != -1:
            self.last_move = (None, None)

        # Check if the move has won the local board
        if self.check_board(self.board[big_row][big_col], self.current_player):
            self.meta_board[big_row][big_col] = self.current_player
            if self.check_board(self.meta_board, self.current_player):
                self.winner = self.current_player

        # Switch players
        self.current_player = 1 - self.current_player

    def check_winner(self):
        if self.winner != -1:
            print(f"Player {self.winner} has won the game!!!")
            return self.winner
        return -1
    
    def is_terminal(self):
        if self.last_move != (None, None):
            for small_row in range(3):
                for small_col in range(3):
                    if self.board[self.last_move[0]][self.last_move[1]][small_row][small_col] == -1:
                        return False
        else:
            for large_row in range(3):
                for large_col in range(3):
                    if self.meta_board[large_row][large_col] == -1:
                        for small_row in range(3):
                            for small_col in range(3):
                                if self.board[large_row][large_col][small_row][small_col] == -1:
                                    return False
        return True

    def payoff(self):
        if self.winner == 0:
            return 1
        elif self.winner == 1:
            return -1
        else:
            return 0

    # for regular playing
    def input_move(self):
        valid_move = False
        while not valid_move:
            try:
                prev_row = self.last_move[0]
                prev_col = self.last_move[1]
                if self.last_move == (None, None):
                    prev_row = 'ANY'
                    prev_col = 'ANY'
                move = input(f"Player {self.current_player} (0 for O, 1 for X), enter your move as '{prev_row} {prev_col} small_row small_col': ")
                move_components = move.split()
                move_values = [int(x) for x in move_components]
                big_row, big_col, small_row, small_col = move_values
                if self.last_move == (None, None) or (big_row, big_col) == self.last_move:
                    self.make_move(big_row, big_col, small_row, small_col)
                    valid_move = True
                else:
                    print(f'Invalid move! You must play in board ({self.last_move[0]}, {self.last_move[1]})')
            except (ValueError, IndexError):
                print("Invalid input! Please enter four integers separated by spaces.")

    # Reset the game for a new round.
    def reset_game(self):
        self.__init__()

if __name__ == "__main__":
    game = UltimateTicTacToe()
    game.display_board()

    while game.check_winner() == -1:
        game.input_move()
        game.display_board()