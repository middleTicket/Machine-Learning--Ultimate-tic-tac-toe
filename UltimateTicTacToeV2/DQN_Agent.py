import torch
import random
import math
from DQN import DQN
from Constant import *
from Board import Board
from Enviroment import Env
import numpy as np

class DQN_Agent:
    def __init__(self, player = 1, parametes_path = None, train = True, env : Env = None):
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.player = player
        self.shape = player
        self.train = train
        self.env = env
        self.setTrainMode()

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_input (self,events= None, board: Board = None, epoch = 0, train = True) -> tuple:
        actions = self.env.legal_actions(board, board.last_played_place)
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                return random.choice(actions)
        
        board_tensor =  board.toTensor().type(torch.float32)
        action_tensor = torch.from_numpy(np.array(actions))
        action_tensor = action_tensor.type(torch.float32)
        expand_state_tensor = board_tensor.unsqueeze(0).repeat((len(action_tensor),1))
        
        with torch.no_grad():
            Q_values = self.DQN(expand_state_tensor, action_tensor)
        max_index = torch.argmax(Q_values)
        return actions[max_index]

    def get_Actions (self, Boards_tensor: Board, dones, miniBoards) -> torch.tensor:
        actions = []
        
        for i, board in enumerate(Boards_tensor):
            if dones[i].item():
                actions.append((0,0))
            else:
                actions.append(self.get_input(events= None, board= self.env.tensorToState(boardTensor=board,miniBoard= miniBoards[i]), train=False))
        return torch.tensor(actions, dtype=torch.float32)

    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
        res = final + (start - final) * math.exp(-1 * epoch/decay)
        return res
    
    def loadModel (self, file):
        self.model = torch.load(file)
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None):
        return self.get_Action(state)