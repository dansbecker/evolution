import itertools
import numpy as np

from copy import copy
from .game import TicTacToe
from .player import ModelBasedPlayer


class Population:
    def __init__(self, pop_size, model, game_maker, rand_level=.05, evolution_lr=.1):
        self.pop_size = pop_size
        self.model = model
        self.game_maker = game_maker
        self.rand_level = rand_level
        self.evolution_lr = evolution_lr
        self.layer_shapes = [layer.shape for layer in self.model.get_weights()]
        self.n_parameters = sum(layer.size for layer in self.model.get_weights())
        self.weights_centers = np.zeros(self.n_parameters)
        self._create_players()
        self.generation_num = 1
        self.prev_gen_scores = None

    def get_scores(self, opponents='self_play', games_per_matchup=1):
        if opponents == 'self_play':
            matchups_by_index = itertools.combinations(range(self.pop_size), self.game_maker.n_players_per_game)
            rewards = np.zeros(self.pop_size)
            for player_indices in matchups_by_index:
                matchup_players = [self.players[i] for i in player_indices]
                for _ in range(games_per_matchup):
                    game_rewards = self.game_maker.make_game(matchup_players).play()
                    for i, player_id in enumerate(player_indices):
                        rewards[player_id] += game_rewards[i]
        else:
            # The [0] in line below pulls out score for first player, which is assumed to be one we're interested in
            rewards = np.array([sum(self.game_maker.make_game([player, opponent]).play()[0] for _ in range(games_per_matchup)
                                                                                            for opponent in opponents)
                                                                                            for player in self.players])
        return(rewards)

    def _normalize_scores(self, scores):
        # Add small number to scores.std in case all scores are equal
        return (scores-scores.mean()) / (scores.std() + 1e-2)

    def score_and_evolve(self, opponents='self_play', games_per_matchup=1):
        self.prev_gen_scores = self.get_scores(opponents, games_per_matchup)
        normalized_scores = self._normalize_scores(self.prev_gen_scores)
        weight_updates = self.evolution_lr / (self.pop_size * self.rand_level) * np.dot(self.weight_deviations.T, normalized_scores)
        self.weights_centers += weight_updates
        self._create_players()
        self.generation_num += 1

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