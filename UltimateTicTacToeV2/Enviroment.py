from Board import Board 
import numpy as np

class Env:
    def __init__(self, board) -> None:
        self.board = board
        
    def make_move(self, board : Board, action : tuple, player : int):
        if not action:
            return 
        row, col = action
        if board.Board[row][col] == 0:
            board.Board[row][col] = player
            board.last_played_place = board.translate_place_to_otter_inner(action)[1]
            board.last_player = player
            board.check_Inner_Winner_By_Place(board.translate_place_to_otter_inner(action)[0])
            if board.check_Winner_For_Inner_Board(board.miniBoard): #if the mini board has a winner 
                board.endWinner = board.check_Winner_For_Inner_Board(board.miniBoard, True)
            elif not self.legal_actions(board, board.last_played_place):
                board.endWinner = 2

    def next_Board (self, board : Board, action):
        #board is not in final stage
        player = board.last_player * -1
        next_board = board.copy()
        self.make_move(next_board, action, player )
        return next_board

    def legal_actions(self, Board : Board ,lastPlacePlayed = (-1,-1)):

        legal = []
        if lastPlacePlayed == (-1,-1):
            for i in range(3):
                for j in range(3):
                    if Board.check_Inner_Winner_By_Place((i,j)):
                        continue
                    innerBoard = Board.get_Inner_Board_By_Place((i,j))
                    for x in range(3):
                        for y in range(3):
                            if innerBoard[x][y] == 0:
                                legal.append(Board.translate_otter_inner_to_place(((i,j), (x,y))))#  for ((0,0), (0,0)) will add (0,0)
            return legal
        else:
            row, col = lastPlacePlayed
            if Board.check_Inner_Winner_By_Place((row,col)):
                return self.legal_actions(Board, lastPlacePlayed = (-1,-1))
            innerBoard = Board.get_Inner_Board_By_Place((row,col))
            for x in range(3):
                for y in range(3):
                    if innerBoard[x][y] == 0:
                        legal.append(Board.translate_otter_inner_to_place(((row,col), (x,y))))#  for ((0,0), (0,0)) will add (0,0)
            return legal

    def is_legal(self, Board: Board, action, lastPlacePlayed = (-1,-1)):
        legal_actions = self.legal_actions(Board, lastPlacePlayed)
        return action in legal_actions

    def reward (self, board : Board, action = None) -> tuple:

        done = board.check_winner()
        if done:
            if board.endWinner != 2:
                return board.endWinner, True
            else:
                return 0, True
        else:
            if len(self.legal_actions(board, board.last_played_place)) == 0:
                return 0, True
        return 0, False
            

    def tensorToState(self, boardTensor, miniBoard):
        b = Board()
        board = boardTensor[0:81]
        last_played_place = int(boardTensor[81:83][0].item()), int(boardTensor[81:83][1].item())
        b.Board = board.numpy().reshape(9,9)
        b.last_played_place = last_played_place
        b.last_player = b.Board[last_played_place[0]][last_played_place[1]].shape
        b.miniBoard = np.array(miniBoard.numpy().reshape(3,3))
        return b
