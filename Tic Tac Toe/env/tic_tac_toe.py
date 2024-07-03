""" Logics of the game Tic Tac Toe """

import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.player = 1
        self.winner = None
        self.done = False

    def reset(self):
        self.board = np.zeros((3, 3))
        self.player = 1
        self.winner = None
        self.done = False

    def step(self, action):
        if self.done:
            return self.board, 0, self.done

        if self.board[action] != 0:
            return self.board, -10, self.done

        self.board[action] = self.player

        if self.check_winner():
            self.winner = self.player
            self.done = True
            return self.board, 10, self.done

        if np.all(self.board != 0):
            self.done = True
            return self.board, 0, self.done

        self.player = -1 if self.player == 1 else 1
        return self.board, 0, self.done

    def check_winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return True
        return False

    def render(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 1:
                    print('X', end=' ')
                elif self.board[i][j] == -1:
                    print('O', end=' ')
                else:
                    print('_', end=' ')
            print()
        print()