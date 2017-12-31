import numpy as np

from copy import copy
from .game import TicTacToe
from .player import ModelBasedPlayer




class Population:
    def __init__(self, pop_size, model, board_dims, n_players_per_game=2, rand_level=1, evolution_lr=0.01):
        # todo: constructor should accept a Game Factory rather than board_dims and n_players_per_game
        self.pop_size = pop_size
        self.model = model
        self.board_dims = board_dims
        self.n_players_per_game = n_players_per_game
        self.rand_level = rand_level
        self.evolution_lr = evolution_lr
        self.layer_shapes = [layer.shape for layer in self.model.get_weights()]
        self.n_parameters = sum(layer.size for layer in self.model.get_weights())
        self.weights_centers = np.zeros(self.n_parameters)
        self._create_players()

    def get_scores(self, opponents=None, games_per_matchup=1):
        if opponents == None:
            #TODO run round robin tournament
            pass
        else:
            rewards = np.array([sum(TicTacToe([player, opponent], self.board_dims).play()[0] for _ in range(games_per_matchup)
                                                                                              for opponent in opponents)
                                                                                              for player in self.players])
            return(rewards)

    def _normalize_scores(self, scores):
        return (scores-scores.mean()) / scores.std()

    def score_and_evolve(self, opponents=None, games_per_matchup=5):
        scores = self.get_scores(opponents, games_per_matchup)
        normalized_scores = self._normalize_scores(scores)
        weight_updates = self.evolution_lr / (self.pop_size * self.rand_level) * np.dot(self.weight_deviations.T, normalized_scores)
        self.weights_centers += weight_updates
        self._create_players()

    def _create_players(self):
        # TODO: Clean-up for readability (remove for loop?)
        self.players = []
        self.weight_deviations = np.random.rand(self.pop_size, self.n_parameters)
        weights_all_players = self.weights_centers.T + self.weight_deviations
        for i in range(self.pop_size):
            model = copy(self.model)
            model_weights = self._make_weights_correct_shape(weights_all_players[i,:])
            model.set_weights(model_weights)
            self.players.append(ModelBasedPlayer(model))
        return


    def _make_weights_correct_shape(self, weights):
        out = []
        current_loc = 0
        for l_shape in self.layer_shapes:
            l_size = np.product(l_shape)
            out.append(weights[current_loc:(current_loc+l_size)].reshape(l_shape))
            current_loc = current_loc + l_size
        return out