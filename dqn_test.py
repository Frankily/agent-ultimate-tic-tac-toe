import random

from typing import Callable, Any

from game import Game, State
from modeltest import Model, Encoder
from qfl_game import QFLGame, QFLEncoder
from replay import ReplayDB

def epsilon_greedy(eps):
    def select(values):
        if (random.random() < eps):
            return random.choice(range(len(values)))
        else:
            return max(enumerate(values), key=lambda p: p[1])[0]
    return select


class DQN:
    def __init__(self, game: Game, encoder: Encoder):
        """
        Initializes a DQN trainer for the given game.  States will be
        encoded as inputs to the neural networks using the given encoder.

        game -- a game
        encoder -- an encoder for that game
        """
        self._game = game
        self._encoder = encoder
        self._num_actions = len(game.initial_state().get_actions())

        hidden = [20, 20]
        
        self._target = Model(encoder, self._num_actions, hidden)
        self._target.eval()

        self._learning = Model(encoder, self._num_actions, hidden)
        self._learning.train()
        

    def simulate(self, num_games = 65536, games_at_once = 32):
        states = [self._game.initial_state() for i in range(games_at_once)]
        term_count = 0
        wins = 0
        select = epsilon_greedy(0.0)
        
        while term_count < num_games:
            values = self._learning.predict(states)
            actions = [select(qs) for qs in values]
            new_s = [s.successor(a) for s, a in zip(states, actions)]
            terminal = [s.is_terminal() for s in new_s]
            rewards = [(s.payoff() if t else 0) for s, t in zip(new_s, terminal)]
            term_count += sum(1 if curr_t else 0 for curr_t in terminal)
            wins += sum(1 if r == 1 else 0 for r in rewards)
            states = [(self._game.initial_state() if t else s) for s, t in zip(new_s, terminal)]
            
        return wins / term_count

        
    def train(self, replay: ReplayDB, games_at_once=4, epsilon = 0.5, train_interval = 4, train_batch = 32, transfer_interval = 8192, episodes = 100000):
        """
        Trains the networks for this DQN agent.
        """
        states = [self._game.initial_state() for i in range(games_at_once)]
        term_count = 0
        action_count = 0
        next_train = train_interval
        next_xfer = transfer_interval
        select = epsilon_greedy(epsilon)
        
        while term_count < episodes:
            values = self._learning.predict(states)
            actions = [select(qs) for qs in values]
            new_s = [s.successor(a) for s, a in zip(states, actions)]
            terminal = [s.is_terminal() for s in new_s]
            rewards = [(s.payoff() if t else 0) for s, t in zip(new_s, terminal)]
            for s, a, s2, r in zip(states, actions, new_s, rewards):
                replay.add(s, a, s2, r)
            term_count += sum(1 if curr_t else 0 for curr_t in terminal)
            action_count += games_at_once
            states = [self._game.initial_state() if t else s for s, t in zip(new_s, terminal)]
            
            if action_count >= next_train and action_count > train_batch:
                samples = replay.sample(train_batch)
                self._learning.step(samples, self._target)
                next_train += train_interval
                epsilon = max(epsilon * 0.999, 0.1)
                
            if action_count >= next_xfer:
                self._target.copy(self._learning)
                next_xfer += transfer_interval
                print(self.simulate())

                
    def policy(self) -> Callable[[State], Any]:
        """
        Returns a policy that plays the game using the trained network.
        """
        def choose_action(s: State):
            return max(enumerate(self._learning.predict([s])), key=lambda p: p[1])[0]
        return choose_action

    
if __name__ == "__main__":
    game = QFLGame()
    encoder = QFLEncoder(game)
    replay = ReplayDB(10, 5, 100000, game.state_partitioner([10, 5]))
    dqn = DQN(game, encoder)
    dqn.train(replay)
