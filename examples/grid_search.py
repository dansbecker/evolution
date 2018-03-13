from evolution.game import GameMaker, TicTacToe
from evolution.player import StrategyFunctionPlayer
from evolution.population import Population

from keras.models import Model
from keras.layers import Dense, Flatten, Input

from numpy import mean, inf
from random import shuffle
import pandas as pd


example_dims = [3,3]

def get_example_model(board_dims=example_dims, n_players=2, hidden_size=20):
    input_board = Input(board_dims + [n_players])
    flattened_input = Flatten()(input_board)
    hidden_layer_1 = Dense(hidden_size, activation='relu')(flattened_input)
    output_row = Dense(board_dims[0], activation='softmax')(hidden_layer_1)
    output_col = Dense(board_dims[1], activation='softmax')(hidden_layer_1)
    my_model = Model(input_board, [output_row, output_col])
    return(my_model)

def first_open_square_player():
    def first_open_square_strat(board):
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if (board[i,j, :] == 0).all():
                    return (i, j)
        # all squares are full
        return (0, 0)
    return StrategyFunctionPlayer(board_dims=example_dims, move_fn=first_open_square_strat)

def random_open_square_player():
    def random_open_square_strat(board):
        moves = [(x, y) for x in range(board.shape[0]) for y in range(board.shape[1])]
        shuffle(moves)
        for i, j in moves:
            if (board[i, j, :] == 0).all():
                return (i, j)

        # all squares are full
        return (0, 0)
    return StrategyFunctionPlayer(board_dims=example_dims, move_fn=random_open_square_strat)

if __name__ == "__main__":
    n_generations = 50
    n_agents = 50
    games_per_matchup = 20
    max_possible_score = games_per_matchup

    results = []
    for lr in [0.5, 1, 2]:
        for rand_level in [0.1, 0.5]:
            for hidden_size in [25]:
                TicTacToeMaker = GameMaker(TicTacToe, board_dims=example_dims)
                my_model = get_example_model(hidden_size=hidden_size)
                my_pop = Population(n_agents, model=my_model, update_method='grad',
                                    game_maker=TicTacToeMaker, evolution_lr=lr, rand_level=rand_level)
                for i in range(n_generations):
                    if my_pop.prev_gen_scores is None or my_pop.prev_gen_scores.mean() < max_possible_score:
                        my_pop.score_and_evolve(opponents=[random_open_square_player()],
                                                games_per_matchup=games_per_matchup)

                current_results = {'lr': lr, 'rand_level': rand_level, 'hidden_size': hidden_size,
                                           'generations': my_pop.generation_num, 'mean_pop_score': my_pop.prev_gen_scores.mean(),
                                           'best_score_in_pop': my_pop.prev_gen_scores.max()}
                print(current_results)
                results.append(current_results)
    pd.DataFrame(results).to_csv('./optimization_results.csv', index=False)
    print('Done')
