import pytest

from evolution.population import Population
from test_game import first_open_square_player, always_middle_square_player
from fixtures import get_example_model, TicTacToeMaker

def test_scratchwork():
    model = get_example_model()
    my_opponents = [first_open_square_player(), always_middle_square_player()]
    my_pop = Population(30, model=model, game_maker=TicTacToeMaker)
    some_weights_from_player_0_before_evolution = my_pop.players[0].model.get_weights()[0]
    my_pop.score_and_evolve(my_opponents, 10)
    some_weights_from_player_0_after_evolution = my_pop.players[0].model.get_weights()[0]
    assert(not (some_weights_from_player_0_before_evolution == some_weights_from_player_0_after_evolution).all())