""" Logics of the game Tic Tac Toe """

import torch
import numpy as np

class TicTacToe:
    def __init__(self, N=3):
        self.N = N
        self.action_space = N * N
        self.state_space = N * N * 3
        self.reset()

    def reset(self):
        """
        Resets the board and returns the initial state
        """
        self.player = 1
        self.board = np.zeros((self.N, self.N), dtype=int)
        return self.__state

    def step(self, action: int) -> tuple:
        """
        Returns the next state, reward and done
        """
        row = action // self.N
        col = action % self.N

        self.board[row, col] = self.player

        self.player = -self.player

        if self.__check_win(action, -self.player, self.board):
            return self.__state, 1, True

        if np.all(self.board != 0):
            return self.__state, 0, True

        return self.__state, 0, False

    def get_valid_actions(self, board=None):
        """
        Returns a binary array of valid actions
        """
        if board is None:
            board = self.board
        return (board.flatten() == 0).astype(int)

    def __check_win(self, action, player, board):
        row = action // self.N
        col = action % self.N
        
        assert player == board[row, col], "Player and board mismatch"

        return (
            np.sum(board[row, :]) == player * self.N
            or np.sum(board[:, col]) == player * self.N
            or np.sum(np.diag(board)) == player * self.N
            or np.sum(np.diag(np.fliplr(board))) == player * self.N
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

        if self.__check_win(action, player, board):
            return state, 1, True

        if np.all(board != 0):
            return state, 0, True

        return state, 0, False

    def transform(self, board: np.ndarray) -> torch.Tensor:
        state = np.stack(
            [
                board == 1,
                board == -1,
                board == 0,
            ]
        ).astype(np.float32)
        return torch.from_numpy(state)

    def __state__(self, board, player):
        return board * player

    @property
    def __state(self):
       return self.board * self.player 

    def __str__(self):
        _str = '\n-------------\n'
        for i in range(self.N):
            for j in range(self.N):
                if self.board[i][j] == 1:
                    _str += 'X | '
                elif self.board[i][j] == -1:
                    _str += 'O | '
                else:
                    _str += '_ | '
            _str += '\n-------------\n'
        return _str

    def render(self, policy=None, value=None):
        """
        Renders the board with the policy
        """
        if policy is None:
            print(self)
            return
        print(f"Value: {value:.2f}, Policy:\n-------------")
        for i in range(self.N):
            for j in range(self.N):
                print(f"{policy[i * 3 + j]:.2f}", end=" | ")
            print('\n-------------')
        print()
