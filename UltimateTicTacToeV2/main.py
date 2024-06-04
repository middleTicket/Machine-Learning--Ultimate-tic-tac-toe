import pygame
import sys
from Board import Board
from Grapichs import Grapichs
from HumanAgent import HumanAgent
from Enviroment import Env
from RandomAgent import RandomAgent
from MinMaxAgent import MinMaxAgent
from AlphaBetaAgent import AlphaBeta
from DQN_Agent_CNN import DQN_Agent_CNN
from DQN_Agent import DQN_Agent

pygame.init()

# Define some colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (200, 125, 255)
CYAN = (0, 255, 255)
COLOR = (127, 127, 127)

# Screen dimensions
DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600

# Function to display text on the screen
def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, True, color)
    textrect = textobj.get_rect()
    textrect.center = (x, y)
    surface.blit(textobj, textrect)

# Main game function
def main(player1, player2):
    display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption('Ultimate Tic Tac Toe')

    B = Board()
    env = Env(B)
    G = Grapichs(B)

    player = player1
    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
                exit()

        action = player.get_input(events=events, board=B.copy())
        if action:
            sp = 0
            if player.shape == -1:
                sp = 'x'
            else:
                sp = 'o'
            print(f"last move: {B.translate_place_to_otter_inner(action)}, by : {sp}")
            env.make_move(B, action, player.shape)
            if player == player1:
                player = player2
            else:
                player = player1

        G.draw(display)
        pygame.display.update()
        B.check_winner()
        if B.endWinner is not None:
            if B.endWinner == 1:
                finaleWinner = 'o'
            elif B.endWinner == -1:
                finaleWinner = 'x'
            else:
                print(B.miniBoard)
                finaleWinner = 'tie no winner'
                pygame.time.delay(10000)
            print(finaleWinner)
            pygame.time.delay(5000)
            break

    pygame.quit()
    sys.exit()

# Start screen function
def start_screen():
    display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption('Ultimate Tic Tac Toe')

    font = pygame.font.Font(None, 74)
    smallfont = pygame.font.Font(None, 36)
    click = False

    player1 = None
    player2 = None

    while True:
        display.fill(WHITE)
        draw_text('Ultimate Tic Tac Toe', font, BLACK, display, DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 6)

        mx, my = pygame.mouse.get_pos()

        # Player 1 buttons on the right
        draw_text('Select Player 1', smallfont, BLACK, display, DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 4)

        button_1_1 = pygame.Rect(50, 200, 200, 50)
        button_1_2 = pygame.Rect(50, 260, 200, 50)
        button_1_3 = pygame.Rect(50, 320, 200, 50)
        button_1_4 = pygame.Rect(50, 380, 200, 50)
        button_1_5 = pygame.Rect(50, 440, 200, 50)
        button_1_6 = pygame.Rect(50, 500, 200, 50)

        pygame.draw.rect(display, RED, button_1_1)
        pygame.draw.rect(display, GREEN, button_1_2)
        pygame.draw.rect(display, BLUE, button_1_3)
        pygame.draw.rect(display, PURPLE, button_1_4)
        pygame.draw.rect(display, CYAN, button_1_5)
        pygame.draw.rect(display, COLOR, button_1_6)

        draw_text('Human', smallfont, WHITE, display, 150, 225)
        draw_text('Random', smallfont, WHITE, display, 150, 285)
        draw_text('MinMax', smallfont, WHITE, display, 150, 345)
        draw_text('AlphaBeta', smallfont, WHITE, display, 150, 405)
        draw_text('CNN', smallfont, WHITE, display, 150, 465)
        draw_text('DDQN', smallfont, WHITE, display, 150, 525)

        # Player 2 buttons on the left
        draw_text('Select Player 2', smallfont, BLACK, display, 3 * DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 4)

        button_2_1 = pygame.Rect(350, 200, 200, 50)
        button_2_2 = pygame.Rect(350, 260, 200, 50)
        button_2_3 = pygame.Rect(350, 320, 200, 50)
        button_2_4 = pygame.Rect(350, 380, 200, 50)
        button_2_5 = pygame.Rect(350, 440, 200, 50)
        button_2_6 = pygame.Rect(350, 500, 200, 50)

        pygame.draw.rect(display, RED, button_2_1)
        pygame.draw.rect(display, GREEN, button_2_2)
        pygame.draw.rect(display, BLUE, button_2_3)
        pygame.draw.rect(display, PURPLE, button_2_4)
        pygame.draw.rect(display, CYAN, button_2_5)
        pygame.draw.rect(display, COLOR, button_2_6)

        draw_text('Human', smallfont, WHITE, display, 450, 225)
        draw_text('Random', smallfont, WHITE, display, 450, 285)
        draw_text('MinMax', smallfont, WHITE, display, 450, 345)
        draw_text('AlphaBeta', smallfont, WHITE, display, 450, 405)
        draw_text('CNN', smallfont, WHITE, display, 450, 465)
        draw_text('DDQN', smallfont, WHITE, display, 450, 525)

        if click:
            # Player 1 selection
            if button_1_1.collidepoint((mx, my)):
                player1 = HumanAgent(Board(), 1, Env(Board()))
            if button_1_2.collidepoint((mx, my)):
                player1 = RandomAgent(Board(), 1, Env(Board()))
            if button_1_3.collidepoint((mx, my)):
                player1 = MinMaxAgent(Board().copy(), 1, Env(Board()), 1)
            if button_1_4.collidepoint((mx, my)):
                player1 = AlphaBeta(Board().copy(), 1, Env(Board()), 4)
            if button_1_5.collidepoint((mx, my)):
                player1 = DQN_Agent_CNN(1, train=False, parametes_path="Data/best_params_99.pth", env=Env(Board()))
            if button_1_6.collidepoint((mx, my)):
                player1 = DQN_Agent(1, train=False, parametes_path="Data/best_params_22.pth", env=Env(Board()))

            # Player 2 selection
            if button_2_1.collidepoint((mx, my)):
                player2 = HumanAgent(Board(), -1, Env(Board()))
            if button_2_2.collidepoint((mx, my)):
                player2 = RandomAgent(Board(), -1, Env(Board()))
            if button_2_3.collidepoint((mx, my)):
                player2 = MinMaxAgent(Board().copy(), -1, Env(Board()), 3)
            if button_2_4.collidepoint((mx, my)):
                player2 = AlphaBeta(Board().copy(), -1, Env(Board()), 4)
            if button_2_5.collidepoint((mx, my)):
                player2 = DQN_Agent_CNN(-1, train=False, env=Env(Board()))
            if button_2_6.collidepoint((mx, my)):
                player2 = DQN_Agent(-1, train=False, parametes_path="Data/best_params_200.pth", env=Env(Board()))
            # If both players are selected, start the game
            if player1 and player2:
                launch_game(player1, player2)

        click = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    click = True

        pygame.display.update()

def launch_game(player1, player2):
    pygame.display.quit()
    pygame.display.init()
    main(player1, player2)

if __name__ == "__main__":
    start_screen()
