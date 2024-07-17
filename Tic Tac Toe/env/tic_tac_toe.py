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
        row = action // self.N
        col = action % self.N

        self.board[row, col] = self.player

        if self.__check_win(action):
            return self.__state, 1, True

        if np.all(self.board != 0):
            return self.__state, 0, True

        self.player = -self.player

        return self.__state, 0, False

    def get_valid_actions(self):
        """
        Returns a binary array of valid actions
        """
        return (self.board.flatten() == 0).astype(int)

    def __check_win(self, action):
        row = action // self.N
        col = action % self.N
        
        assert self.player == self.board[row, col]

        return (
            np.sum(self.board[row, :]) == self.player * self.N
            or np.sum(self.board[:, col]) == self.player * self.N
            or np.sum(np.diag(self.board)) == self.player * self.N
            or np.sum(np.diag(np.fliplr(self.board))) == self.player * self.N
        )

    @staticmethod
    def transform(board: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            [
                (board == 1).astype(int),
                (board == -1).astype(int),
                (board == 0).astype(int)
            ], dtype=torch.float32
        )

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
