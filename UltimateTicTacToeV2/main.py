import pygame
from Board import Board
from Grapichs import Grapichs
from HumanAgent import HumanAgent
from Enviroment import Env
from RandomAgent import RandomAgent
from MinMaxAgent import MinMaxAgent
from AlphaBetaAgent import AlphaBeta
# from DQN_Agent import DQN_Agent
from Tester import Tester
from DQN_Agent_CNN import DQN_Agent

pygame.init()

def main():
    DISPLAY_WIDTH = 600
    DISPLAY_HEIGHT = 600
    display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption('Ultimate Tic Tac Toe')
    pygame.display.set_caption("Tic Tac Toe")


    B = Board()
    env = Env(B)
    G = Grapichs(B)
    ###################################################### CHANGE PLAYERS #################################
    # player1 = HumanAgent(B, 1, env)#o = 1
    player1 = RandomAgent(B, 1, env)#o = 1
    # player1 = MinMaxAgent(B.copy(), 1, env, 1)#o = 1
    # player1 = AlphaBeta(B.copy(), 1, env, 5)
    # player1 = DQN_Agent(1,train=False, parametes_path= "Data/best_params_99.pth",env=env)#Switch between CNN and DQN and vice versa (siwtch the imported model)
    # player2 = HumanAgent(B, -1, env)#x = -1
    player2 = RandomAgent(B, -1, env)#x = -1
    # player2 = MinMaxAgent(B.copy(), -1, env,3)#x = -1
    # player2 = AlphaBeta(B.copy(), -1, env, 2)   
    # player2 = DQN_Agent(-1,train=False, env=env)
    

    player = player1
    # Game loop
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
            # pygame.time.delay(5000)
            if player == player1:
                player = player2
            else:
                player = player1
            
        # Draw the board
        G.draw(display)
        B.check_winner()
        if B.endWinner:
            if B.endWinner == 1:
                finaleWinner = 'o'
            elif B.endWinner == -1:
                finaleWinner = 'x'
            else: 
                print(B.miniBoard)  
                finaleWinner = 'tie no winner'
                pygame.time.delay(10000)
            print(finaleWinner)
            # pygame.time.delay(1000)
            break
            

    # Quit Pygame
    pygame.quit()



if __name__ == "__main__":
    main()