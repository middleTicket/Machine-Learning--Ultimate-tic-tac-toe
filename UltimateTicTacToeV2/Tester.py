from RandomAgent import RandomAgent
from Board import Board
from Enviroment import Env
from MinMaxAgent import MinMaxAgent
from AlphaBetaAgent import AlphaBeta
from DQN_Agent_CNN import DQN_Agent


class Tester:
    def __init__(self, env : Env, player1, player2) -> None:
        self.env = env
        if player1.shape == 1:
            self.player1 = player1# O, 1
            self.player2 = player2# X, -1
        else:
            self.player2 = player1# O, 1
            self.player1 = player2# X, -1
            

    def test (self, games_num):
        env = self.env
        player = self.player1
        player1_win = 0
        player2_win = 0
        games = 0
        B = Board()
        while games < games_num:
            print(f'game num = {games}', end='\r', flush=True)
            action = player.get_input(events=None, board=B.copy(), train = False)
            env.make_move(B,action, player.shape)
            player = self.switchPlayers(player)
            B.check_winner()
            if B.endWinner == -1 or B.endWinner == 1:
                
                score = B.endWinner
                if score > 0:
                    player1_win += 1
                elif score < 0:
                    player2_win += 1
                # env.state = env.get_init_state()
                games += 1
                player = self.player1
                B = Board()
            elif B.endWinner or env.legal_actions(B, B.last_played_place) == []:

                B = Board()
                games += 1
                player = self.player1
        if self.player1.shape == 1:
            return player1_win, player2_win        
        else:
            return player2_win, player1_win
    
    def switchPlayers(self, player):
        if player == self.player1:
            return self.player2
        else:
            return self.player1

    def __call__(self, games_num):
        return self.test(games_num)



def main():
    B = Board()
    env = Env(B)
    # player1 = AlphaBeta(B, 1, env, 3)
    # player1 = MinMaxAgent(B, 1, env, 3)
    # player1 = RandomAgent(B, 1, env)#o = 1
    player1 = DQN_Agent(1,train=False, parametes_path= "Data/best_params_99.pth",env=env)#Switch between CNN and DQN and vice versa
    
    # player1 = DQN_Agent(1,train=False, parametes_path= "Data/best_params_3.pth",env=env)
    # player2 = HumanAgent(B, -1, env)#x = -1
    # player2 = RandomAgent(B, -1, env)#x = -1
    # player2 = MinMaxAgent(B, -1, env,3)#x = -1
    player2 = AlphaBeta(B, -1, env, 3)   
    # player2 = DQN_Agent(-1,train=False, parametes_path="Data/best_params_200copy.pth" ,env=env)
    tester = Tester(env,player1, player2)
    o_wins, x_wins = tester(50)
    print(f"O wins is {o_wins * 100 / (o_wins + x_wins)} precent")
    print(f"o wins {o_wins}, x wins {x_wins}")




if __name__ == '__main__':
    main()