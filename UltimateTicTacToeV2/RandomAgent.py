import pygame
from Board import Board
from Enviroment import Env
import random

class RandomAgent:
    def __init__(self, board, shape : int, env : Env):
        self.board = board
        self.shape = shape
        self.env = env

    def get_input(self, events = None, board : Board = None, train = None):
        legal_actions = self.env.legal_actions(board, board.last_played_place)
        if not legal_actions:
            return None
        return random.choice(legal_actions)
        