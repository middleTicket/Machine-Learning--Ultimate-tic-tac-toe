import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

gamma = 0.99

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1,8,3,3)
        self.fc1 = nn.Linear(8 * 3 * 3 + 2, 64)
        self.fc2 = nn.Linear(64,1)
        self.MSELoss = nn.MSELoss()
        self.device = torch.device('cpu')
        
    def forward(self, states , action):
        states = states[:,0:81]
        states = states.view(-1,1, 9, 9)
        x = F.relu(self.conv1(states))
        x = x.view(-1, 8 * 3 * 3)         
        x = torch.cat((x,action), dim=1)
        x = F.relu(self.fc1(x))            
        x = F.relu(self.fc2(x))             
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


    def __call__(self, states, action) -> torch.Any:
        return self.forward(states, action)
