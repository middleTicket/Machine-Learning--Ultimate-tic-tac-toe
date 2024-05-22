import pygame 
import numpy as np
import torch

class Board:

    def __init__(self, startValue = -1, otherBoard = None, miniBoard = None ) -> None:#startValue is x, x = -1
        
        # if not otherBoard:
        if otherBoard is None or np.all(otherBoard == 0):
            self.Board = np.zeros((9, 9))
            self.endWinner = None
            self.miniBoard =np.zeros((3,3))#created for checking the winner of each
            #inner board
            self.last_player = startValue
            self.last_played_place = (-1,-1)
        else:
            self.Board = np.array(otherBoard)
            self.miniBoard = miniBoard
            self.endWinner = None
            self.last_player = startValue
            self.last_played_place = (-1,-1)

    def get_Inner_Board_By_Place(self, OuterPlace : tuple):
        row, col = OuterPlace
        return self.Board[row*3:(row+1)*3, col*3:(col+1)*3]
        
    def check_Winner_For_Inner_Board(self, board, get_winner = False):# recive a board & return a winner or if there is a winner 
        if not get_winner:   
            for row in board:
                if row[0] == row[1] == row[2] and row[0] != 0:
                    return True
            # Check columns for a winner
            for col in range(3):
                if board[0][col] == board[1][col] == board[2][col] and board[0][col] != 0:
                    return True
            # Check diagonals for a winner
            if board[0][0] == board[1][1] == board[2][2] and board[0][0] != 0:
                return True
            if board[0][2] == board[1][1] == board[2][0] and board[0][2] != 0:
                return True
            
            # If no winner yet, return False
            return False
        else:
            for row in board:
                if row[0] == row[1] == row[2] and row[0] != 0:
                    return row[0]
            # Check columns for a winner
            for col in range(3):
                if board[0][col] == board[1][col] == board[2][col] and board[0][col] != 0:
                    return board[0][col]
            # Check diagonals for a winner
            if board[0][0] == board[1][1] == board[2][2] and board[0][0] != 0:
                return board[0][0]
            if board[0][2] == board[1][1] == board[2][0] and board[0][2] != 0:
                return board[0][2]
            
            # If no winner yet, and every place has a player in it than tie
            if np.count_nonzero(board) == 0:
                return 2

            return 0

    def check_Inner_Winner_By_Place(self, OtterPlace : tuple): #outer being (0,0) - (2,2)
        # Check rows for a winner
        board = self.get_Inner_Board_By_Place(OtterPlace)
        for row in board:
            if row[0] == row[1] == row[2] and row[0] != 0:
                self.miniBoard[OtterPlace[0]][OtterPlace[1]] = row[0]
                return True 
        # Check columns for a winner
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] and board[0][col] != 0:
                self.miniBoard[OtterPlace[0]][OtterPlace[1]] = board[0][col]
                return True 
        # Check diagonals for a winner
        if board[0][0] == board[1][1] == board[2][2] and board[0][0] != 0:
            self.miniBoard[OtterPlace[0]][OtterPlace[1]] = board[0][0]
            return True 
        if board[0][2] == board[1][1] == board[2][0] and board[0][2] != 0:
            self.miniBoard[OtterPlace[0]][OtterPlace[1]] = board[0][2]
            return True 

        # If no winner yet, return False
        self.miniBoard[OtterPlace[0]][OtterPlace[1]] = 0

        return False

    def copy(self):
        board = self.Board.copy()
        miniboard = self.miniBoard.copy()
        new_board = Board(startValue=self.last_player, otherBoard=board, miniBoard=miniboard)
        new_board.endWinner = self.endWinner
        new_board.last_played_place = self.last_played_place
        return new_board

    def translate_place_to_otter_inner(self, place : tuple):
        if not place:
            return None
        row, col = place
        outer_row = row // 3
        outer_col = col // 3
        inner_row = row % 3
        inner_col = col % 3
        outer_coordinates = (outer_row, outer_col)
        inner_coordinates = (inner_row, inner_col)
        return outer_coordinates, inner_coordinates
      
    def translate_otter_inner_to_place(self, outer_inner_coordinates: tuple):# ((outerX, oterY), (inner..))
        outer_row, outer_col = outer_inner_coordinates[0]
        inner_row, inner_col = outer_inner_coordinates[1]
        row = outer_row * 3 + inner_row
        col = outer_col * 3 + inner_col
        return row, col
    
    def check_winner(self):
        # after every move you check the miniboard for Victory and update miniboard. 1 - O, -1 - X  , 0 - play
        # function return 1, -1, 0
        for i in range(3):
            for j in range(3):
                if self.miniBoard[i][j] == 0:
                    self.check_Inner_Winner_By_Place((i,j))

        if self.check_Winner_For_Inner_Board(self.miniBoard):
            self.endWinner = self.check_Winner_For_Inner_Board(self.miniBoard, True)
            return True
        return False

    
    def toTensor(self):
        
        boardTensor = torch.tensor(self.Board.reshape(-1))
        last_played_place_tensor = torch.tensor(self.last_played_place)
        return torch.cat((boardTensor, last_played_place_tensor)).to(torch.float32)

