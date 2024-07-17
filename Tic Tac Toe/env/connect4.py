""" Logics of the game Connect 4 """

import torch
import numpy as np

class Connect4:
    def __init__(self, N=6, M=7):
        self.N = N
        self.M = M
        self.action_space = M
        self.state_space = N * M * 3
        self.reset()

    def reset(self):
        """
        Resets the board and returns the initial state
        """
        self.player = 1
        self.board = np.zeros((self.N, self.M), dtype=int)
        return self.__state

    def step(self, action: int) -> tuple:
        """
        Returns the next state, reward and done
        """
        row = max(np.where(self.board[:, action] == 0)[0])
        self.board[row, action] = self.player

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
        return (board[0, :] == 0).astype(int)

    def __check_win(self, action, player, board):
        # Check horizontal locations for win
        for c in range(self.M - 3):
            for r in range(self.N):
                if board[r][c] == player and board[r][c + 1] == player and board[r][c + 2] == player and board[r][c + 3] == player:
                    return True

        # Check vertical locations for win
        for c in range(self.M):
            for r in range(self.N - 3):
                if board[r][c] == player and board[r + 1][c] == player and board[r + 2][c] == player and board[r + 3][c] == player:
                    return True

        # Check positively sloped diagonals
        for c in range(self.M - 3):
            for r in range(self.N - 3):
                if board[r][c] == player and board[r + 1][c + 1] == player and board[r + 2][c + 2] == player and board[r + 3][c + 3] == player:
                    return True

        # Check negatively sloped diagonals
        for c in range(self.M - 3):
            for r in range(3, self.N):
                if board[r][c] == player and board[r - 1][c + 1] == player and board[r - 2][c + 2] == player and board[r - 3][c + 3] == player:
                    return True

        return False

    def _step(self, board, action, player=None):
        """
        The step function without changing the class variables
        """
        if player is None:
            player = self.player
        board = board.copy()
        row = max(np.where(board[:, action] == 0)[0])
        board[row, action] = player

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
        _str = '\n--------------------------\n'
        for i in range(self.N):
            for j in range(self.M):
                _str += f"{self.board[i, j]} | "
            _str += '\n--------------------------\n'
        return _str

    def render(self, policy=None, value=None):
        """
        Renders the board with the policy
        """
        if policy is None:
            print(self)
            return
        print(f"Value: {value:.2f}, Policy:\n-------------")
        print(policy)
