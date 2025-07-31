import numpy as np
from keras import layers
import gymnasium as gym # maj vers gymnasium
import tensorflow as tf
import random
from collections import deque

import os
os.environ["OMP_NUM_THREADS"] = "14" 
tf.config.threading.set_intra_op_parallelism_threads(14)
tf.config.threading.set_inter_op_parallelism_threads(14)

# ------------------------------------------------------------------------

env_name = "MountainCar-v0"  # changer le nom du gym
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99
gamma = 0.99
batch_size = 32
memory_size = 100000
episodes = 500


# ------------------------------------------------------------------------

env = gym.make(env_name) #, render_mode="human") # render_mode pour avoir un visu avec pygame
state_shape = env.observation_space.shape[0]
action_shape = env.action_space.n
state_shape, action_shape

# ------------------------------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoint/dqn_model.weights.h5',
        monitor='loss',
        save_best_only=True,
        save_weights_only=True
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# ------------------------------------------------------------------------

def create_q_model():
    model = tf.keras.Sequential(
        [
            layers.Dense(64, input_shape=(state_shape, )),    # augmentation des neurones 32 -> 64
            layers.Dense(64),              # augmentation des neurones 32 -> 64 = pas de relu
            layers.Dense(action_shape, activation='linear')
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse') # ajout du lr
    return model

q_model = create_q_model()
target_model = create_q_model()
target_model.set_weights(q_model.get_weights())


# ------------------------------------------------------------------------

memory = deque(maxlen=memory_size)

def store_transition(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


# Echantillonnage d'un batch de transition
def sample_batch():
    batch = random.sample(memory, batch_size) # 32
    state, action, reward, next_state, done = map(np.asarray, zip(*batch))
    return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
# politique epsilon-greedy
def epsilon_greedy_policy(state, epsilon):
  if np.random.random() < epsilon:
    return np.random.choice(action_shape)
  else:
    q_values = q_model.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0])


# ------------------------------------------------------------------------

def train_step():
    if len(memory) < batch_size:
        return
    state, action, reward, next_state, done = sample_batch()

    # Forward propagation
    next_q_values = target_model.predict(next_state, verbose=0)
    max_next_q_values = np.max(next_q_values, axis=1)


    target_q_values = q_model.predict(state, verbose=0)
    for i, action in enumerate(action):
        target_q_values[i][action] = reward[i] if done[i] else reward[i] + gamma * max_next_q_values[i]

    q_model.fit(state, target_q_values, verbose=0, callbacks=callbacks)


# ------------------------------------------------------------------------

reward_history = []


for episode in range(episodes):
    state, _ = env.reset()  # maj vers gymnasium 
    total_reward = 0
    done = False


    while not done:
        action = epsilon_greedy_policy(state, epsilon)
        
        next_state, reward, terminated, truncated, _ = env.step(action)  # maj vers gymnasium
        
        done = terminated or truncated # done true en fonction de truncated ou terminated
        # print(f"State: {state}\t|\tAction: {action}\t|\tReward: {reward}\t|\tNext State: {next_state}\t|\tDone: {done}")

        store_transition(state, action, reward, next_state, done)
        total_reward += reward

        state = next_state
        train_step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 10 == 0:
        target_model.set_weights(q_model.get_weights())
    
    reward_history.append(total_reward)
    print(f"\n\n\t\t\t----- Episode: {episode}\t|\tReward: {total_reward}\t|\tEpsilon: {epsilon:.2f} -----\n\n")
