""" Logics of the game Tic Tac Toe """

import torch
import numpy as np

class TicTacToe:
    def __init__(self, N=3):
        self.N = N
        self.action_space = N * N
        self.state_space = N * N * 3
        self.board = np.zeros((N, N), dtype=int)
        self.player = 1

    def reset(self):
        """
        Resets the board and returns the initial state
        """
        self.board = np.zeros((self.N, self.N), dtype=int)
        return self.__state

    def step(self, action: int) -> tuple:
        """
        Returns the next state, reward and done
        """
        
        self.board, reward, done = self._step(self.board, action)
        self.player = -self.player

        return self.__state, reward, done

    def get_valid_actions(self):
        """
        Returns a binary array of valid actions
        """
        return (self.board.flatten() == 0).astype(int)

    def __check_win(self, action, player):
        row = action // self.N
        col = action % self.N
        
        assert player == self.board[row, col], "Player and board mismatch"

        return (
            np.sum(self.board[row, :]) == player * self.N
            or np.sum(self.board[:, col]) == player * self.N
            or np.sum(np.diag(self.board)) == player * self.N
            or np.sum(np.diag(np.fliplr(self.board))) == player * self.N
        )

    def _step(self, board, action, player=None):
        """
        The step function without changing the class variables
        """
        if player is None:
            player = self.player
        board = board.copy()
        row = action // self.N
        col = action % self.N

        board[row, col] = player

        state = self.__state__(board, player)

        if self.__check_win(board, action, player):
            return state, 1, True

        if np.all(board != 0):
            return state, 0, True

        return state, 0, False

    def transform(board: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            [
                (board == 1).astype(int),
                (board == -1).astype(int),
                (board == 0).astype(int)
            ], dtype=torch.float32
        )

    def __state__(self, board, player):
        return board * player

    @property
    def __state(self):
       return self.board * self.player 

    def __str__(self):
        _str = ''
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 1:
                    _str += 'X '
                elif self.board[i][j] == -1:
                    _str += 'O '
                else:
                    _str += '_ '
            _str += '\n'
        return _str

    def render(self):
        # TODO: Implement a better render function
        print(self)
