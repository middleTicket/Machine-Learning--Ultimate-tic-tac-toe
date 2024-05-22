import numpy as np


board = np.zeros((9,9))


for i in range(3):
            row_values = board[i*3:(i+1)*3, :].flatten()
            col_values = board[:, i*3:(i+1)*3].flatten()
            