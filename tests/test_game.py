import numpy as np
import pytest

from evolution.game import TicTacToe
from evolution.player import ModelBasedPlayer, StrategyFunctionPlayer

def always_middle_square_player():
    return StrategyFunctionPlayer(board_dims=(3,3), move_fn=lambda _: (1,1))

def first_open_square_player():
    def first_open_square_strat(board):
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if (board[i,j, :] == 0).all():
                    return (i, j)
        # all squares are full
        return (0, 0)
    return StrategyFunctionPlayer(board_dims=(3,3), move_fn=first_open_square_strat)


def test_illegal_move_penalized_1():
    game = TicTacToe([always_middle_square_player(), first_open_square_player()], board_dims=(3,3))
    game_rewards = game.play()
    assert(game_rewards == [-1, 0])
    assert(game.moves_played == 2)

def test_illegal_move_penalized_2():
    game = TicTacToe([first_open_square_player(), always_middle_square_player()], board_dims=(3,3))
    game_rewards = game.play()
    assert(game_rewards == [0, -1])
    assert(game.moves_played == 3)

def test_win_diag():
    game = TicTacToe([first_open_square_player(), first_open_square_player()], board_dims=(3,3))
    game_rewards = game.play()
    assert(game_rewards == [1, -1])
    assert(game.moves_played == 7)

def test_win_column_small():
    game = TicTacToe([first_open_square_player(), first_open_square_player()], board_dims=(2, 2))
    game_rewards = game.play()
    assert(game_rewards == [1, -1])
    assert(game.moves_played == 3)

def test_moves_possible():
    game1 = TicTacToe([first_open_square_player(), first_open_square_player()], board_dims=(2, 2))
    game1.board[0,0,0] = 1
    assert(game1.moves_possible())
    game1.board[0, 1, 1] = 1
    assert (game1.moves_possible())
    game1.board[1,0,0] = 1
    assert (game1.moves_possible())
    game1.board[1,1,1] = 1
    assert (not game1.moves_possible())
    game2 = TicTacToe([first_open_square_player(), first_open_square_player()], board_dims=(3, 3))
    game2.board[0:2,0,0] = 1
    game2.board[0:2, 1, 1] = 1
    assert (game2.moves_possible())
    game2.board[0:2, 2, 1] = 1
    assert (game2.moves_possible())
    game2.board[2, :, 1] = 1
    assert (not game2.moves_possible())
    game3 = TicTacToe([first_open_square_player(), first_open_square_player()], board_dims=(4, 4))
    game3.board[0:3, 0, 1] = 1
    game3.board[0:3, 1, 1] = 1
    assert (game3.moves_possible())

def test_check_move_legality():
    game = TicTacToe([first_open_square_player(), first_open_square_player()], board_dims=(3, 3))
    game.board[0,0,0] = 1
    game.board[1,1,1] = 1
    game.board[2,0,0] = 1
    assert (not game._check_move_legality((0,0)))
    assert (game._check_move_legality((0, 1)))
    assert (game._check_move_legality((1, 0)))
    assert (not game._check_move_legality((2, 0)))
    assert (game._check_move_legality((2, 2)))