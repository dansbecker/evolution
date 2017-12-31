import numpy as np
import pytest

from evolution.player import ModelBasedPlayer, StrategyFunctionPlayer
from evolution.game import TicTacToe
from evolution.population import Population
from fixtures import get_example_model, TicTacToeMaker

def test_moves_right_dims():
    n_players_per_game=2
    #todo: test a range of board sizes, requires being able to tie model shape to game board shape
    for board_size in [3]:
        board_dims = (board_size, board_size)
        board_storage_dims = list(board_dims) + [n_players_per_game]
        my_pop = Population(pop_size=2, model=get_example_model(), game_maker=TicTacToeMaker)
        my_player = my_pop.players[0]
        moves = my_player.get_move(np.zeros(board_storage_dims), player_id=0)
        assert(len(moves) == 2)
        assert(all(move in range(board_size) for move in moves))

