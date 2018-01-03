"""
Game of Reversi
"""

from six import StringIO
import sys
import gym
from gym import spaces
import numpy as np
from gym import error
from gym.utils import seeding

def make_random_policy(np_random):
    def random_policy(state, player_color):
        possible_places = ReversiEnv.get_possible_actions(state, player_color)
        # No places left
        if len(possible_places) == 0:
            return d**2 + 1
        a = np_random.randint(len(possible_places))
        return possible_places[a]
    return random_policy

class ReversiEnv(gym.Env):
    """
    Reversi environment. Play against a fixed opponent.
    """
    BLACK = 0
    WHITE = 1
    metadata = {"render.modes": ["ansi","human"]}

    def __init__(self, player_color, opponent, observation_type, illegal_place_mode, board_size):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: An opponent policy
            observation_type: State encoding
            illegal_place_mode: What to do when the agent makes an illegal place. Choices: 'raise' or 'lose'
            board_size: size of the Reversi board
        """
        assert isinstance(board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size

        colormap = {
            'black': ReversiEnv.BLACK,
            'white': ReversiEnv.WHITE,
        }
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error("player_color must be 'black' or 'white', not {}".format(player_color))

        self.opponent = opponent

        assert observation_type in ['numpy3c']
        self.observation_type = observation_type

        assert illegal_place_mode in ['lose', 'raise']
        self.illegal_place_mode = illegal_place_mode

        if self.observation_type != 'numpy3c':
            raise error.Error('Unsupported observation type: {}'.format(self.observation_type))

        # One action for each board position and resign and pass
        self.action_space = spaces.Discrete(self.board_size ** 2 + 2)
        observation = self.reset()
        self.observation_space = spaces.Box(np.zeros(observation.shape), np.ones(observation.shape))

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == 'random':
                self.opponent_policy = make_random_policy(self.np_random)
                print("################################################################")
            else:
                raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent

        return [seed]

    def _reset(self):
        # init board setting
        self.state = np.zeros((3, self.board_size, self.board_size))
        self.state[2, :, :] = 1.0
        self.state[2, 3:5, 3:5] = 0
        self.state[0, 4, 3] = 1
        self.state[0, 3, 4] = 1
        self.state[1, 3, 3] = 1
        self.state[1, 4, 4] = 1
        self.to_play = ReversiEnv.BLACK
        self.possible_actions = ReversiEnv.get_possible_actions(self.state, self.to_play)
        self.done = False

        # Let the opponent play if it's not the agent's turn
        if self.player_color != self.to_play:
            a = self.opponent_policy(self.state)
            ReversiEnv.make_place(self.state, a, ReversiEnv.BLACK)
            self.to_play = ReversiEnv.WHITE
        return self.state

    def _step(self, action):
        color = action[1]
        action = action[0]
        
        assert self.to_play == self.player_color
        # If already terminal, then don't do anything
        if self.done:   # 如果已经结束了
            return self.state, 0., True, {'state': self.state}
        if color == 0:  #  黑色棋子是 0
            if ReversiEnv.pass_place(self.board_size, action):
                pass
            elif ReversiEnv.resign_place(self.board_size, action):
                return self.state, -1, True, {'state': self.state}
            elif not ReversiEnv.valid_place(self.state, action, self.player_color):
                if self.illegal_place_mode == 'raise':
                    raise
                elif self.illegal_place_mode == 'lose':
                    # Automatic loss on illegal place
                    self.done = True
                    return self.state, -1., True, {'state': self.state}
                else:
                    raise error.Error('Unsupported illegal place action: {}'.format(self.illegal_place_mode))
            else:
                ReversiEnv.make_place(self.state, action, self.player_color)
                self.possible_actions = ReversiEnv.get_possible_actions(self.state, 1)

        else:       # # Opponent play  白色棋子 是 1
            if ReversiEnv.pass_place(self.board_size, action):
                pass
            elif ReversiEnv.resign_place(self.board_size, action):
                return self.state, 1, True, {'state': self.state}
            elif not ReversiEnv.valid_place(self.state, action, 1 - self.player_color):
                if self.illegal_place_mode == 'raise':
                    raise
                elif self.illegal_place_mode == 'lose':
                    # Automatic loss on illegal place
                    self.done = True
                    return self.state, 1., True, {'state': self.state}
                else:
                    raise error.Error('Unsupported illegal place action: {}'.format(self.illegal_place_mode))
            else:
                ReversiEnv.make_place(self.state, action , 1 - self.player_color)
                self.possible_actions = ReversiEnv.get_possible_actions(self.state, 0 )


        reward = ReversiEnv.game_finished(self.state)
        if self.player_color == ReversiEnv.WHITE:
            reward = - reward
        self.done = reward != 0
        return self.state, reward, self.done, {'state': self.state}

    # def _reset_opponent(self):
    #     if self.opponent == 'random':
    #         self.opponent_policy = random_policy
    #     else:
    #         raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))

    def _render(self, mode='human', close=False):
        if close:
            return
        board = self.state
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        outfile.write(' ' * 7)
        for j in range(board.shape[1]):
            outfile.write(' ' +  str(j + 1) + '  | ')
        outfile.write('\n')
        outfile.write(' ' * 5)
        outfile.write('-' * (board.shape[1] * 6 - 1))
        outfile.write('\n')
        for i in range(board.shape[1]):
            outfile.write(' ' +  str(i + 1) + '  |')
            for j in range(board.shape[1]):
                if board[2, i, j] == 1:
                    outfile.write('  O  ')
                elif board[0, i, j] == 1:
                    outfile.write('  B  ')
                else:
                    outfile.write('  W  ')
                outfile.write('|')
            outfile.write('\n')
            outfile.write(' ' )
            outfile.write('-' * (board.shape[1] * 7 - 1))
            outfile.write('\n')

        if mode != 'human':
            return outfile

    # @staticmethod
    # def pass_place(board_size, action):
    #     return action == board_size ** 2

    @staticmethod
    def resign_place(board_size, action):
        return action == board_size ** 2

    @staticmethod
    def pass_place(board_size, action):
        return action == board_size ** 2 + 1

    @staticmethod
    def get_possible_actions(board, player_color):
        actions=[]
        d = board.shape[-1]
        opponent_color = 1 - player_color
        for pos_x in range(d):
            for pos_y in range(d):
                if (board[2, pos_x, pos_y]==0):
                    continue
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if(dx == 0 and dy == 0):
                            continue
                        nx = pos_x + dx
                        ny = pos_y + dy
                        n = 0
                        if (nx not in range(d) or ny not in range(d)):
                            continue
                        while(board[opponent_color, nx, ny] == 1):
                            tmp_nx = nx + dx
                            tmp_ny = ny + dy
                            if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                                break
                            n += 1
                            nx += dx
                            ny += dy
                        if(n > 0 and board[player_color, nx, ny] == 1):
                            actions.append(pos_x*8+pos_y)
        if len(actions)==0:
            actions = [d**2 + 1]
        return actions

    @staticmethod
    def valid_reverse_opponent(board, coords, player_color):
        '''
        check whether there is any reversible places
        这里意思应该是 判断这里是否有 翻转的 棋子。
        '''
        d = board.shape[-1]
        opponent_color = 1 - player_color
        pos_x = coords[0]
        pos_y = coords[1]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if(dx == 0 and dy == 0):
                    continue
                nx = pos_x + dx
                ny = pos_y + dy
                n = 0
                if (nx not in range(d) or ny not in range(d)):
                    continue
                while(board[opponent_color, nx, ny] == 1):
                    tmp_nx = nx + dx
                    tmp_ny = ny + dy
                    if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                        break
                    n += 1
                    nx += dx
                    ny += dy
                if(n > 0 and board[player_color, nx, ny] == 1):
                    return True
        return False

    @staticmethod
    def valid_place(board, action, player_color):
        coords = ReversiEnv.action_to_coordinate(board, action)
        # check whether there is any empty places
        if board[2, coords[0], coords[1]] == 1:
            # check whether there is any reversible places
            if ReversiEnv.valid_reverse_opponent(board, coords, player_color):
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def make_place(board, action, player_color):
        coords = ReversiEnv.action_to_coordinate(board, action)

        d = board.shape[-1]
        opponent_color = 1 - player_color
        pos_x = coords[0]
        pos_y = coords[1]

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if(dx == 0 and dy == 0):
                    continue
                nx = pos_x + dx
                ny = pos_y + dy
                n = 0
                if (nx not in range(d) or ny not in range(d)):
                    continue
                while(board[opponent_color, nx, ny] == 1):
                    tmp_nx = nx + dx
                    tmp_ny = ny + dy
                    if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                        break
                    n += 1
                    nx += dx
                    ny += dy
                if(n > 0 and board[player_color, nx, ny] == 1):
                    nx = pos_x + dx
                    ny = pos_y + dy
                    while(board[opponent_color, nx, ny] == 1):
                        board[2, nx, ny] = 0
                        board[player_color, nx, ny] = 1
                        board[opponent_color, nx, ny] = 0
                        nx += dx
                        ny += dy
                    board[2, pos_x, pos_y] = 0
                    board[player_color, pos_x, pos_y] = 1
                    board[opponent_color, pos_x, pos_y] = 0
        return board

    @staticmethod
    def coordinate_to_action(board, coords):
        return coords[0] * board.shape[-1] + coords[1]

    @staticmethod
    def action_to_coordinate(board, action):
        return action // board.shape[-1], action % board.shape[-1]

    @staticmethod
    def game_finished(board):
        # Returns 1 if player 1 wins, -1 if player 2 wins and 0 otherwise
        d = board.shape[-1]

        player_score_x, player_score_y = np.where(board[0, :, :] == 1)
        player_score = len(player_score_x)
        opponent_score_x, opponent_score_y = np.where(board[1, :, :] == 1)
        opponent_score = len(opponent_score_x)
        if player_score == 0:
            return -1
        elif opponent_score == 0:
            return 1
        else:
            free_x, free_y = np.where(board[2, :, :] == 1)
            if free_x.size == 0:
                if player_score > (d**2)/2:
                    return 1
                elif player_score == (d**2)/2:
                    return 1
                else:
                    return -1
            else:
                return 0
        return 0
