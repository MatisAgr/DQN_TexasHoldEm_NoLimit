import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100_000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.model = self._build_model(lr)

    def _build_model(self, lr):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
        return model

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def act(self, state, legal_actions=None):
        if legal_actions is None or len(legal_actions) == 0:
            return random.randrange(self.action_size)
        legal_actions = list(legal_actions)
        if np.random.rand() < self.epsilon:
            return random.choice(legal_actions)
        q = self.model.predict(np.array([state]), verbose=0)[0]
        mask = np.full(self.action_size, -np.inf)
        mask[legal_actions] = q[legal_actions]
        if np.all(np.isneginf(mask)):
            return random.choice(legal_actions)
        action = int(np.argmax(mask))
        if action not in legal_actions:
            return random.choice(legal_actions)
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        for s, a, r, s_, done in minibatch:
            target = self.model.predict(np.array([s]), verbose=0)[0]
            if done:
                target[a] = r
            else:
                t = self.model.predict(np.array([s_]), verbose=0)[0]
                target[a] = r + self.gamma * np.amax(t)
            states.append(s)
            targets.append(target)
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay