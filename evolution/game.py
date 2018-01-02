import numpy as np

class BoardGame:
    def __init__(self, players, board_dims):
        self.players = players
        self.n_players = len(players)
        self.board_dims = board_dims
        self.board_storage_dims = list(board_dims) + [self.n_players]
        self.board = np.zeros(self.board_storage_dims)
        self.moves_played = 0
        return

    def play(self):
        rewards = [0 for player in self.players]
        while self.moves_possible():
            for player_id, player in enumerate(self.players):
                move = player.get_move(self.board, player_id)
                if not self._check_move_legality(move):
                    rewards[player_id] = -1
                    return rewards
                self._apply_move(move, player_id)
                if self._check_for_win(player_id):
                    rewards = [-1 for player in self.players]
                    rewards[player_id] = 1
                    return rewards
        return(rewards)


class TicTacToe(BoardGame):
    def _check_for_win(self, player_id):
        for row in range(self.board_storage_dims[0]):
            if (self.board[row, :, player_id] == 1).all():
                return True
        for col in range(self.board_storage_dims[1]):
            if (self.board[:, col, player_id] == 1).all():
                return True
        if (self.board[:, :, player_id].diagonal() == 1).all():
            return True
        if (self.board[:,::-1,player_id].diagonal() == 1).all(): # other diagonal
            return True
        return False

    def _apply_move(self, move, player_id):
        self.board[move[0], move[1], player_id] = 1
        self.moves_played+=1
        return

    def moves_possible(self):
        return (self.board.sum(2)==0).any()

    def _check_move_legality(self, move):
        return (self.board[move[0], move[1], :] == 0).all()


class GameMaker:
    def __init__(self, game_class, board_dims, n_players_per_game=2):
        self.game_class = game_class
        self.board_dims = board_dims
        self.n_players_per_game = n_players_per_game

    def make_game(self, players):
        return self.game_class(players, self.board_dims)

