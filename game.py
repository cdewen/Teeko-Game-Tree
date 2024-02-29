import random
import numpy as np
import copy

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def up(self, k, i, j):
        state = copy.deepcopy(k)
        if i - 1 >= 0 and state[i - 1][j] == ' ':
            state[i][j], state[i - 1][j] = state[i - 1][j], state[i][j]
            return state

    def down(self, k, i, j):
        state = copy.deepcopy(k)
        if i + 1 < len(state) and state[i + 1][j] == ' ':
            state[i][j], state[i + 1][j] = state[i + 1][j], state[i][j]
            return state

    def left(self, k, i, j):
        state = copy.deepcopy(k)
        if j - 1 >= 0 and state[i][j - 1] == ' ':
            state[i][j], state[i][j - 1] = state[i][j - 1], state[i][j]
            return state

    def right(self, k, i, j):
        state = copy.deepcopy(k)
        if j + 1 < len(state) and state[i][j + 1] == ' ':
            state[i][j], state[i][j + 1] = state[i][j + 1], state[i][j]
            return state

    def upleft(self, k, i, j):
        state = copy.deepcopy(k)
        if i - 1 >= 0 and j - 1 >= 0 and state[i - 1][j - 1] == ' ':
            state[i][j], state[i - 1][j - 1] = state[i - 1][j - 1], state[i][j]
            return state

    def upright(self, k, i, j):
        state = copy.deepcopy(k)
        if i - 1 >= 0 and j + 1 < len(state) and state[i - 1][j + 1] == ' ':
            state[i][j], state[i - 1][j + 1] = state[i - 1][j + 1], state[i][j]
            return state

    def downleft(self, k, i, j):
        state = copy.deepcopy(k)
        if i + 1 < len(state) and j - 1 >= 0 and state[i + 1][j - 1] == ' ':
            state[i][j], state[i + 1][j - 1] = state[i + 1][j - 1], state[i][j]
            return state

    def downright(self, k, i, j):
        state = copy.deepcopy(k)
        if i + 1 < len(state) and j + 1 < len(state) and state[i + 1][j + 1] == ' ':
            state[i][j], state[i + 1][j + 1] = state[i + 1][j + 1], state[i][j]
            return state

    def get_succ(self,state):
        
        succ = []

        drop_phase = True

        numB = sum((i.count('b') for i in state))
        numR = sum((i.count('r') for i in state))
        if numB >= 4 and numR >= 4:
            drop_phase = False

        if not drop_phase:
            for row in range(len(state)):
                for col in range(len(state)):
                    if state[row][col] == self.my_piece:
                        succ.insert(0, self.up(state, row, col)) # (row-1)(col)
                        succ.insert(1, self.down(state, row, col)) # (row+1)(col)
                        succ.insert(2, self.left(state, row, col)) # (row)(col-1)
                        succ.insert(3, self.right(state, row, col)) #
                        succ.insert(4, self.upleft(state, row, col))
                        succ.insert(5, self.upright(state, row, col))
                        succ.insert(6, self.downleft(state, row, col))
                        succ.insert(7, self.downright(state, row, col))
            return list(filter(None, succ))
        #must consider the drop phase
        else:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == ' ':
                        state[i][j] = self.my_piece
                        succ.append(copy.deepcopy(state))
                        state[i][j] = ' '

        return succ
        
    def heuristic_game_value(self, state):
        
        best_self = 0
        best_opp = 0

        # check horizontal in a row
        for row in state:
            for i in range(2):
                if row[i] == self.my_piece:
                    for j in range(4):
                        if row[j] == self.my_piece:
                            best_self += 1
                if row[i] == self.opp:
                    for j in range(4):
                        if row[j] == self.opp:
                            best_opp += 1

        # check vertical in a row
        for col in range(5):
            for i in range(2):
                if state[i][col] == self.my_piece:
                    for j in range(4):
                        if state[j][col] == self.my_piece:
                            best_self += 1
                if state[i][col] == self.opp:
                    for j in range(4):
                        if state[j][col] == self.opp:
                            best_opp += 1

        # check \ diagonal in a row
        for i in range(2):
            for j in range(2):
                if state[i][j] == self.my_piece:
                    for k in range(4):
                        if state[i+k][j+k] == self.my_piece:
                            best_self += 1
                if state[i][j] == self.opp:
                    for k in range(4):
                        if state[i+k][j+k] == self.opp:
                            best_opp += 1

        # check / diagonal in a row
        for i in range(3,5):
            for j in range(2):
                if state[i][j] == self.my_piece:
                    for k in range(4):
                        if state[i-k][j+k] == self.my_piece:
                            best_self += 1
                if state[i][j] == self.opp:
                    for k in range(4):
                        if state[i-k][j+k] == self.opp:
                            best_opp += 1
        # check box in a row
        for i in range(4):
            for j in range(4):
                if state[i][j] == self.my_piece:
                    if state[i+1][j] == self.my_piece: best_self += 1
                    if state[i][j+1] == self.my_piece: best_self += 1
                    if state[i+1][j+1] == self.my_piece: best_self += 1

                if state[i][j] == self.opp:
                    if state[i+1][j] == self.opp: best_opp += 1
                    if state[i][j+1] == self.opp: best_opp += 1
                    if state[i+1][j+1] == self.opp: best_opp += 1

        x = np.array([best_self, best_opp])
        softmax = np.exp(x)/sum(np.exp(x))

        if best_self > best_opp:
            return softmax[0]
        elif best_self < best_opp:
            return -softmax[1]
        else:
            return 0
        
    def max_value(self, state, depth):
        if self.game_value(state) != 0:
            return self.game_value(state)
        elif depth == 0:
            return self.heuristic_game_value(state)
        #if its my turn     
        if depth % 2 == 0:
            succ = [self.max_value(succ, depth-1) for succ in self.get_succ(state)]
            return max(succ)
        #if its opp turn
        else:
            succ = [self.max_value(succ, depth-1) for succ in self.get_succ(state)]
            return min(succ)
        
        
    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """


        drop_phase = True  # TODO: detect drop phase

        numB = sum((i.count('b') for i in state))
        numR = sum((i.count('r') for i in state))
        if numB >= 4 and numR >= 4:
            drop_phase = False
        if not drop_phase:
            # TODO: choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!


            move = []
            bstate = []
            succ = self.get_succ(state)
            best = -1
            for i in range(len(succ)):
                val = self.max_value(succ[i], 4)
                if val > best:
                    best = val
                    bstate = succ[i]
            for i in range(5):
                for j in range(5):
                    if state[i][j] != bstate[i][j]:
                        if state[i][j] == self.my_piece:
                            move.insert(1,(i,j))
                        else:
                            move.insert(0,(i,j))
            print(move)
            return move
            


        # select an unoccupied space randomly
        # TODO: implement a minimax algorithm to play better

        move = []
        bstate = []
        succ = self.get_succ(state)
        best = -1
        for i in range(len(succ)):
            val = self.max_value(succ[i], 4)
            if val > best:
                best = val
                bstate = succ[i]
        for i in range(5):
            for j in range(5):
                if state[i][j] != bstate[i][j]:
                    move.append((i,j))
        return move


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # check \ diagonal wins
        for i in range(2):
            for j in range(2):
                if state[i][j] != ' ' and state[i][j] == state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3]:
                    return 1 if state[i][j]==self.my_piece else -1
        # check / diagonal wins
        for i in range(3,5):
            for j in range(2):
                if state[i][j] != ' ' and state[i][j] == state[i-1][j-1] == state[i-2][j-2] == state[i-3][j-3]:
                    return 1 if state[i][j]==self.my_piece else -1
        # check box wins
        for i in range(4):
            for j in range(4):
                if state[i][j] != ' ' and state[i][j] == state[i+1][j] == state[i][j+1] == state[i+1][j+1]:
                    return 1 if state[i][j]==self.my_piece else -1

        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
