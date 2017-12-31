import keras
from keras.models import Model
from keras.layers import *

from evolution.game import GameMaker, TicTacToe

TicTacToeMaker = GameMaker(TicTacToe, board_dims=(3,3))

def get_example_model(board_dims=[3, 3], n_players=2):
    input_board = Input(board_dims + [n_players])
    flattened_input = Flatten()(input_board)
    hidden_layer = Dense(100, activation='relu')(flattened_input)
    output_row = Dense(board_dims[0], activation='softmax')(hidden_layer)
    output_col = Dense(board_dims[1], activation='softmax')(hidden_layer)
    my_model = Model(input_board, [output_row, output_col])
    return(my_model)