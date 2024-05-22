import pygame, numpy
from Board import Board

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)



DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600

class Grapichs:
            
        def __init__(self, board : Board):
            self.board = board
            self.inner_board_size = DISPLAY_WIDTH // 9

        
        def draw(self, display):
            inner_board_size = DISPLAY_WIDTH // 9
            # Draw the outer board
            pygame.draw.rect(display, BLACK, (0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT), 2)
            for i in range(3):
                for j in range(3):
                    pygame.draw.rect(display, WHITE, (i*inner_board_size*3, j*inner_board_size*3, inner_board_size*3, inner_board_size*3), 2)
            

             # Draw the inner boards
            for i in range(3):
                for j in range(3):
                    inner_board = self.board.get_Inner_Board_By_Place((i,j))
                    for x in range(3):
                        for y in range(3):
                            value = inner_board[x][y]
                            color = BLACK
                            if value == -1:# x = -1
                                color = RED
                            elif value == 1:# o = 1
                                color = BLUE
                            elif value == 0:
                                color = WHITE
                            else:
                                pass
                            pygame.draw.circle(display, color, ((j*3+y)*self.inner_board_size+self.inner_board_size//2, (i*3+x)*self.inner_board_size+self.inner_board_size//2), self.inner_board_size//2-10)
            pygame.display.update()
