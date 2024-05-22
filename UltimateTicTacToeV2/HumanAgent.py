import pygame
from Board import Board
from Enviroment import Env

class HumanAgent:
    def __init__(self, board : Board, shape : int, env : Env):
        self.board = board
        self.shape = shape
        self.env = env

    def get_input(self, events = None, board = None):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                cell_size = 600 // 9  # Adjust this based on your actual cell size
                y = pos[0] // cell_size
                x = pos[1] // cell_size
                if not self.env.is_legal(board, (x,y), board.last_played_place):
                    print("invalid action")
                    return None
                return x, y
        return None