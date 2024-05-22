import numpy as np
import torch
from collections import deque
from Board import Board
import random

capacity = 100_000

class ReplayBuffer:
    
    def __init__(self, capacity = capacity, path = None) -> None:
        if path:
            self.buffer = torch.load(path).buffer
        else:
            self.buffer = deque(maxlen = capacity)

    def push(self, board : Board,action, reward, next_board : Board, done):
        self.buffer.append((board.toTensor(), torch.from_numpy(np.array(action)).to(dtype=torch.float32), 
                            torch.tensor(reward, dtype=torch.float32), next_board.toTensor(), 
                            torch.tensor(done, dtype=torch.float32), torch.tensor(board.miniBoard.reshape(-1),dtype=torch.float32)))
        
    def sample(self, batch_size):
        if (batch_size > self.__len__()):
            batch_size = self.__len__()
        board_t_and_last_played_tensor,action_t, reward_t,next_board_t, done_t, miniboard = zip(*random.sample(self.buffer, batch_size))
        boards = torch.vstack(board_t_and_last_played_tensor) 
        actions = torch.vstack(action_t)
        rewards = torch.vstack(reward_t)
        next_boards = torch.vstack(next_board_t)
        done = torch.tensor(done_t).long().reshape(-1,1)
        miniboard = torch.vstack(miniboard)
        return boards, actions, rewards, next_boards, done, miniboard
    
    
    
    def __len__(self):
        return len(self.buffer)