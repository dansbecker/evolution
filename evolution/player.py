import numpy as np

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda


class Player:
    pass

class StrategyFunctionPlayer(Player):
    def __init__(self, board_dims, move_fn):
        self.move_fn = move_fn
        self.board_dims = board_dims

    def get_move(self, board, player_id):
        # TODO: Take advantage of player_id so player knows which player they are
        return self.move_fn(board)


class ModelBasedPlayer(Player):
    def __init__(self, model):
        self.model = model

    def _get_move_probs(self, board):
        board_in_batch_of_1 = np.array([board])
        move_probs = self.model.predict(board_in_batch_of_1)
        return move_probs

    def _simulate_move_from_distribs(self, move_probs):
        move_cdfs = [pr.cumsum() for pr in move_probs]
        random_nums = np.random.uniform(size=len(move_cdfs))
        # line below compares rand vals to cdf thresholds. Returns first index below rand val. Returns last index if r is above all cdf thresholds
        moves = (min(np.argwhere((cdf - r) > 0), default=[cdf.shape[0]-1]) for cdf, r in zip(move_cdfs, random_nums))
        # min in the line above returns an array. Convert to tuple of raw integers
        moves = tuple(arr[0] for arr in moves)
        return moves

    def get_move(self, board, player_id):
        # TODO: Take advantage of player_id so player knows which player they are
        move_probs = self._get_move_probs(board)
        move = self._simulate_move_from_distribs(move_probs)
        return move


