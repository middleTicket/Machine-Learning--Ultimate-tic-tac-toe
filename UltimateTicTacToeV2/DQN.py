import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

#input size without mini board
input_size = 85 #(9x9 board + lastPlayedPlace(x,y)) + action(x,y) (81 + 2) + 2 = 85
layer1 = 64
output_size = 1 #Q(Board, action)
gamma = 0.99

class DQN(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.linear1 = nn.Linear(input_size, layer1)
        self.output = nn.Linear(layer1, output_size)
        self.MSELoss = nn.MSELoss()

    def forward (self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.output(x)
        return x
    
    def loss (self, Q_value, rewards, Q_next_Values, Dones ):
        Q_new = rewards + gamma * Q_next_Values * (1- Dones)
        return self.MSELoss(Q_value, Q_new)
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path))

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy (self):
        return copy.deepcopy(self)

    def __call__(self, states, actions):
        state_action = torch.cat((states,actions), dim=1)
        return self.forward(state_action)