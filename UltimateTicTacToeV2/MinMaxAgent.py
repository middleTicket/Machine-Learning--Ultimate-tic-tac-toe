from Enviroment import Env
from Board import Board
import numpy as np



class MinMaxAgent:
    
    def __init__(self, board : Board ,player : int, env : Env,maxDepth = 5):
        self.board = board
        self.player = player
        self.shape = player
        self.maxDepth = maxDepth
        self.env = env

    def heuristic_ultimate_tic_tac_toe(self,board : Board, player : int):
        score = 0

        if board.endWinner == self.player:
            return 1000
        elif board.endWinner == -1 * self.player:
            return -1000
        elif board.endWinner == 2:
            return 0
        
        # Evaluate each local board
        for i in range(3):
            for j in range(3):
                local_board = board.get_Inner_Board_By_Place((i,j))
                score += self.evaluate_local_board(local_board, player)

        # Evaluate the global board
        score += 5 * self.evaluate_local_board(board.miniBoard, player)

        return score

    def evaluate_local_board(self, local_board, player : int):
        score = 0

        # Check rows and columns in the local board
        for i in range(3):
            row_values = local_board[i, :]
            col_values = local_board[:, i]

            # Check rows
            score += self.evaluate_line(row_values, player)

            # Check columns
            score += self.evaluate_line(col_values, player)

        # Check diagonals in the local board
        diag1_values = np.diag(local_board)
        diag2_values = np.diag(np.fliplr(local_board))

        # Check diagonals
        score += self.evaluate_line(diag1_values, player)
        score += self.evaluate_line(diag2_values, player)

        return score

    def evaluate_line(self, line, player : int):
    # Count occurrences of player and empty spaces in the line
        player_count = np.count_nonzero(line == player)
        empty_count = np.count_nonzero(line == 0)

        # If there are two player pieces and an empty space, it's a potential win
        if player_count == 2 and empty_count == 1:
            return 1
        # If there are two opponent pieces and an empty space, it's a potential block
        elif player_count == 0 and empty_count == 1:
            return -1
        else:
            return 0

    def get_input(self, events, train ,board: Board):
        _, move = self.MinMax(board.copy(), self.player, self.maxDepth, board.last_played_place)
        return move

    def MinMax(self, board : Board, player : int, maxDepth : int, lastPlacePlayed : tuple = (-1,-1)):

        board.check_winner()

        if board.endWinner or maxDepth <= 0:
            return self.heuristic_ultimate_tic_tac_toe(board, self.player), None
        
        if self.player == player:
            best_value = -float('inf')
            if lastPlacePlayed != (-1,-1):
                LegalMoves = self.env.legal_actions(board, self.board.translate_place_to_otter_inner(lastPlacePlayed)[1])#self.board is ok because it's only being used for the mathemathical functions 
            else:
                LegalMoves = self.env.legal_actions(board, (-1,-1))
            best_move = None
            for move in LegalMoves:
                
                board_copy = board.copy()
                self.env.make_move(board_copy, move, player)
                value, _= self.MinMax(board_copy, -1 * player , maxDepth - 1, self.board.translate_place_to_otter_inner(move)[1])
                if value > best_value:
                    best_value = value
                    best_move = move
            return best_value, best_move

        else:
            best_value = float('inf')
            if lastPlacePlayed != (-1,-1):
                LegalMoves = self.env.legal_actions(board, self.board.translate_place_to_otter_inner(lastPlacePlayed)[1])
            else:
                LegalMoves = self.env.legal_actions(board, (-1,-1))
            best_move = None
            
            for move in LegalMoves:
                
                board_copy = board.copy()
                self.env.make_move(board_copy, move, player)
                value, _= self.MinMax(board_copy, -1 * player, maxDepth - 1, self.board.translate_place_to_otter_inner(move)[1])
                if value < best_value:
                    best_value = value
                    best_move = move
            return best_value, best_move
        

