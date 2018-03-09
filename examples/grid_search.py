from evolution.game import GameMaker, TicTacToe
from evolution.player import StrategyFunctionPlayer
from evolution.population import Population

from keras.models import Model
from keras.layers import Dense, Flatten, Input

from numpy import mean, inf




def get_example_model(board_dims=[3,3], n_players=2, hidden_size=20):
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
    return StrategyFunctionPlayer(board_dims=(3,3), move_fn=first_open_square_strat)

if __name__ == "__main__":
    for lr in [0.1]:
        for rand_level in [0.01, 0.1]:
            for hidden_size in [20]:
                TicTacToeMaker = GameMaker(TicTacToe, board_dims=(3, 3))
                my_model = get_example_model(hidden_size=hidden_size)
                my_pop = Population(100, model=my_model, update_method='grad',
                                    game_maker=TicTacToeMaker, evolution_lr=lr, rand_level=rand_level)
                for i in range(51):
                    my_pop.score_and_evolve(opponents=[first_open_square_player()], games_per_matchup=10)
                    if ((i % 25) == 0):
                        print("LR: " + str(lr) + "\t rand level: " + str(rand_level) + "\t hidden size: " + str(hidden_size) +
                          "\t Generations: " +str(i) + "\t Mean Recent Score: " + str(my_pop.prev_gen_scores.mean()) +
                          "\t Best Recent Score: " + str(my_pop.prev_gen_scores.max())
                          )
    print('Done')
