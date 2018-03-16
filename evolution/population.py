import itertools
from keras.models import clone_model
import numpy as np
import pathos.pools as pp
from time import time



from .game import TicTacToe
from .player import ModelBasedPlayer


class Population:
    def __init__(self, pop_size, model, game_maker, update_method='grad', rand_level=.1, evolution_lr=1, use_weights_of_passed_model=False):
        print("Initializing population")
        self.pop_size = pop_size
        self.model = model
        self.game_maker = game_maker
        self.update_method = update_method
        self.rand_level = rand_level
        self.evolution_lr = evolution_lr
        self.layer_shapes = [layer.shape for layer in self.model.get_weights()]
        self.n_parameters = sum(layer.size for layer in self.model.get_weights())
        if use_weights_of_passed_model:
            self._set_centers_from_model(model)
        else:
            self.weights_centers = np.zeros(self.n_parameters)
        self.generation_num = 0
        self.players = [ModelBasedPlayer(clone_model(self.model)) for _ in range(pop_size)]
        self._update_player_weights()
        self.player_scores = np.zeros(self.pop_size)
        return

    def set_scores(self, opponents='self_play', games_per_matchup=1):
        if opponents == 'self_play':
            matchups_by_index = itertools.combinations(range(self.pop_size), self.game_maker.n_players_per_game)
            self.player_scores = np.zeros(self.pop_size)
            for id_set in matchups_by_index:
                matchup_players = [self.players[i] for i in id_set]
                for _ in range(games_per_matchup):
                    game_rewards = self.game_maker.make_game(matchup_players).play()
                    for i, player_id in enumerate(id_set):
                        self.player_scores[player_id] += game_rewards[i]
        else:
            # The [0] in line below pulls out score for first player, which is assumed to be one we're interested in
            self.player_scores = np.array([sum(self.game_maker.make_game([player, opponent]).play()[0] for _ in range(games_per_matchup)
                                                                                            for opponent in opponents)
                                                                                            for player in self.players])

    def print_scores_summary(self):
        current_results = {'generations': self.generation_num,
                           'min_pop_score': self.player_scores.min(),
                           'mean_pop_score': self.player_scores.mean(),
                           'best_score_in_pop': self.player_scores.max(),
                           'std_score_in_pop': self.player_scores.std()}
        print(current_results)
        return

    def save_best_model(self, fpath):
        best_player_id = np.argmax(self.player_scores)
        best_player = self.players[best_player_id]
        best_player.model.save(fpath, include_optimizer=False)
        return

    def score_and_evolve(self, opponents='self_play', games_per_matchup=1, n_gens=1, verbosity=5):
        print('Evolving')
        a=time()
        for g in range(n_gens):
            self.set_scores(opponents, games_per_matchup)
            self._update_weights_centers()
            self._update_player_weights()
            self.generation_num += 1
            if (self.generation_num % verbosity == 0):
                self.print_scores_summary()
        print(time()-a)
        return


    def _set_centers_from_model(self, model):
        weights = model.get_weights()
        self.weights_centers = np.hstack([l.flatten() for l in weights])

    def _make_weights_correct_shape(self, weights):
        out = []
        current_loc = 0
        for l_shape in self.layer_shapes:
            l_size = np.product(l_shape)
            out.append(weights[current_loc:(current_loc+l_size)].reshape(l_shape))
            current_loc = current_loc + l_size
        return out

    def _normalize_scores(self, scores):
        # Add small number to scores.std in case all scores are equal
        return (scores-scores.mean()) / (scores.std() + 1e-2)

    def _update_player_weights(self, keep_center_as_new_player=True):
        self.weight_deviations = np.random.rand(self.pop_size, self.n_parameters)
        weights_all_players = self.weights_centers + self.weight_deviations

        for i in range(self.pop_size):
            model_weights = self._make_weights_correct_shape(weights_all_players[i,:])
            self.players[i].model.set_weights(model_weights)

        if keep_center_as_new_player:
            model_weights = self._make_weights_correct_shape(self.weights_centers)
            self.players[0].model.set_weights(model_weights)
        return

    def _update_weights_centers(self):
        if self.update_method == 'grad':
            normalized_scores = self._normalize_scores(self.player_scores)
            weight_updates = self.evolution_lr / (self.pop_size * self.rand_level) * np.dot(self.weight_deviations.T, normalized_scores)
            self.weights_centers += weight_updates
        elif self.update_method == 'best':
            best_player_id = np.argmax(self.player_scores)
            best_weights_in_layers = self.players[best_player_id].model.get_weights()
            best_weights_flattened = np.hstack(l.flatten() for l in best_weights_in_layers)
            self.weights_centers = best_weights_flattened
        elif self.update_method == 'pass':
            pass
        else:
            assert NotImplementedError('Population tried to use an unsupported update_method')
        return

