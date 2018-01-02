import pytest

from evolution.population import Population
from fixtures import get_example_model, TicTacToeMaker
from numpy.random import seed

from test_game import first_open_square_player, always_middle_square_player

def test_weights_change_in_evolution():
    seed(0)
    model = get_example_model(board_dims=[2,2])
    my_pop = Population(2, model=model, game_maker=SmallTicTacToeMaker)
    player_weights_before = my_pop.players[0].model.get_weights()
    my_pop.score_and_evolve()
    player_weights_after = my_pop.players[0].model.get_weights()
    for layer_num in range(len(my_pop.layer_shapes)):
        assert((player_weights_before[layer_num] != player_weights_after[layer_num]).all())

#def test_weight_centers_move_towards_better_player():
#    seed(0)
#    model = get_example_model(board_dims=[2,2])
#    my_pop = Population(2, model=model, game_maker=TicTacToeMaker)
#    all_weights_before = [p[i].model.get_weights() for p in my_pop.players]
#    weight_centers_before =
#    better_player_before =
#    my_pop.score_and_evolve()
#    weight_centers_after =
#    assert weight centers have moved along gradient
