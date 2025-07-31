import rlcard
import numpy as np

class PokerEnv:
    def __init__(self, seed=42):
        self.env = rlcard.make('no-limit-holdem', config={'seed': seed})
        self.state_size = len(self.env.reset()[0]['obs'])
        self.action_size = self.env.num_actions

    def reset(self):
        state, _ = self.env.reset()
        return state['obs'], state['legal_actions']

    def step(self, action):
        next_state, _ = self.env.step(action)
        done = self.env.is_over()
        reward = self.env.get_payoffs()[0] if done else 0
        return next_state['obs'], reward, done, next_state['legal_actions']