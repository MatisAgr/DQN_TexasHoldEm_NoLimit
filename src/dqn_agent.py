# configuration de l'agent DQN

import numpy as np
import tensorflow as tf
from keras import layers
import random
from collections import deque
from typing import List, Tuple, Optional
import os
import datetime
from config import config

class DQNAgent:
    
    def __init__(self, state_size: int, num_actions: int, learning_rate: float = None,
                 epsilon: float = None, epsilon_min: float = None, epsilon_decay: float = None,
                 gamma: float = None, memory_size: int = None):
        

        self.state_size = state_size
        self.num_actions = num_actions
        
        # paramettre de train
        self.learning_rate = learning_rate if learning_rate is not None else config.DQN.LEARNING_RATE
        self.epsilon = epsilon if epsilon is not None else config.DQN.EPSILON_START
        self.epsilon_min = epsilon_min if epsilon_min is not None else config.DQN.EPSILON_MIN
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else config.DQN.EPSILON_DECAY
        self.gamma = gamma if gamma is not None else config.DQN.GAMMA
        self.batch_size = config.DQN.BATCH_SIZE
        
        # mémoire
        memory_size = memory_size if memory_size is not None else config.DQN.MEMORY_SIZE
        self.memory = deque(maxlen=memory_size)
        
        # regrouper les stats de train
        self.training_stats = {
            'losses': [],
            'rewards': [],
            'episode_lengths': [],
            'q_values': [],
            'epsilon_history': []
        }
        
        # TensorBoard writer personnalisé
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_log_dir = os.path.join(config.PATHS.TENSORBOARD_LOG_DIR, f"dqn_training_{timestamp}")
        self.tensorboard_writer = tf.summary.create_file_writer(self.tensorboard_log_dir)
        
        # copie des poids
        self.q_model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.q_model.get_weights())
    
    # creation du modele
    def create_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential()
        
        # couche de input
        model.add(layers.Dense(config.DQN.HIDDEN_LAYERS[0], activation='relu', input_shape=(self.state_size,)))
        model.add(layers.Dropout(config.DQN.DROPOUT_RATE))
        
        # couche cachée
        for i, layer_size in enumerate(config.DQN.HIDDEN_LAYERS[1:], 1):
            model.add(layers.Dense(layer_size, activation='relu'))
            # Moins de dropout dans les dernières couches
            dropout_rate = config.DQN.DROPOUT_RATE * (0.7 if i >= len(config.DQN.HIDDEN_LAYERS) - 1 else 1.0)
            model.add(layers.Dropout(dropout_rate))

        # couche de sortie
        model.add(layers.Dense(self.num_actions, activation='linear'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model
    
    # callbacks
    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        # un dossier par session de train
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_log_dir = os.path.join(config.PATHS.TENSORBOARD_LOG_DIR, f"training_{timestamp}")
        
        return [
            # TensorBoard pour le monitoring
            tf.keras.callbacks.TensorBoard(
                log_dir=tensorboard_log_dir,
                update_freq='epoch'  # maj par epoch
            ),
            # sauvegarde le meilleur modele tpis mes 
            tf.keras.callbacks.ModelCheckpoint(
                filepath=config.PATHS.FINAL_MODEL,
                monitor='loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            ),
            # reduit le learning rate si pas d'amelioration
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=20,
                min_lr=1e-7,
                verbose=0
            ),
            # early stop si pas d'amelioration
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,  # 10 derniere epochs sans amelioration
                restore_best_weights=True,
                verbose=0
            ),
        ]
    
    # stocke une transition dans la memoire de replay
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))
    
    # retourne toutes les actions legales (elles le sont toutes mdr)
    def get_legal_actions(self) -> List[int]:
        return list(range(self.num_actions))
    
    # choisit une action selon epsilon-greedy
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            # EXPLORATION : action aléatoire
            return np.random.choice(self.get_legal_actions())
        else:
            # EXPLOITATION : meilleure action selon les Q-values
            q_values = self.q_model.predict(state[np.newaxis], verbose=0)[0]
            return np.argmax(q_values)

    # mettre à jour l'epsilon pour la politique epsilon-greedy
    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # reduction exponentielle de l'epsilon 
        
        # Enregistrer l'historique d'epsilon
        self.training_stats['epsilon_history'].append(self.epsilon)
        if len(self.training_stats['epsilon_history']) > 1000:
            self.training_stats['epsilon_history'].pop(0)
    
    # retourne les Q-values pour un etat donné
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return self.q_model.predict(state[np.newaxis], verbose=0)[0]
    
    # echantillonne un batch de transitions de la mémoire
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch)) # decomposition des transitions
        return states, actions, rewards, next_states, dones
    
    # effectue un pas d'entraînement du modele DQN
    def train_step(self, batch_size: int = None) -> Optional[float]:
        # utilise la taille de batch par defaut si aucune n'est specifiee
        if batch_size is None:
            batch_size = self.batch_size
        
        # pas assez d'experiences en memoire pour faire un batch complet
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        
        # Calcul des Q-values cibles
        # predit les q-values pour tous les etats suivants avec le reseau cible
        next_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1) # Q(s', a') pour les états suivants
        
        target_q_values = self.q_model.predict(states, verbose=0) # Q(s, a) pour les etats actuels
        
        # met a jour seulement les q-values des actions qui ont ete prises
        for i in range(batch_size):
            if dones[i]:
                # episode termine : q-value = reward seulement (pas de futur)
                target_q_values[i][actions[i]] = rewards[i] # Q(s, a) = r si l'etat suivant est terminal
            else:
                # episode continue : q-value = reward + valeur future escomptee
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * max_next_q_values[i] # Q(s, a) = r + gamma * max_a' Q(s', a')
        
        callbacks = self.get_callbacks()
        history = self.q_model.fit(states, target_q_values, verbose=1, epochs=1, callbacks=callbacks)
        
        loss = history.history['loss'][0]
        
        # Enregistrer les statistiques
        self.training_stats['losses'].append(loss) # enregistrer la perte
        if len(self.training_stats['losses']) > 1000:  # garder seulement les 1000 dernières
            self.training_stats['losses'].pop(0) # garder la mémoire légère
            
        return loss
    
    # Met à jour l'epsilon pour la politique epsilon-greedy
    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # reduction exponentielle de l'epsilon 
        
        # Enregistrer l'historique d'epsilon
        self.training_stats['epsilon_history'].append(self.epsilon)
        if len(self.training_stats['epsilon_history']) > 1000:
            self.training_stats['epsilon_history'].pop(0)
    
    # enregistre les statistiques d'un épisode
    # episode_reward: récompense totale de l'épisode
    # episode_length: longueur de l'épisode
    # avg_q_value: q-value moyenne de l'épisode
    def record_episode_stats(self, episode_reward: float, episode_length: int, avg_q_value: float = None) -> None:
        # éviter les doublons
        if not self.training_stats['rewards'] or self.training_stats['rewards'][-1] != episode_reward:
            self.training_stats['rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(episode_length)
            
            if avg_q_value is not None:
                self.training_stats['q_values'].append(avg_q_value)
            
            # 1000 dernières entrées
            for key in ['rewards', 'episode_lengths', 'q_values']:
                if len(self.training_stats[key]) > 1000:
                    self.training_stats[key].pop(0)
    
    # retourne les statistiques d'entrainement
    def get_training_stats(self) -> dict:
        return self.training_stats.copy()
    
    # affiche les statistiques d'entrainement
    def print_training_stats(self, episode: int, window: int = 100) -> None:
        if not self.training_stats['rewards']:
            return
            
        recent_rewards = self.training_stats['rewards'][-window:]
        recent_losses = self.training_stats['losses'][-window:]
        recent_lengths = self.training_stats['episode_lengths'][-window:]
        
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        avg_loss = np.mean(recent_losses) if recent_losses else 0
        avg_length = np.mean(recent_lengths) if recent_lengths else 0
        win_rate = len([r for r in recent_rewards if r > 0]) / len(recent_rewards) * 100 if recent_rewards else 0
        
        print(f"\n{'='*80}")
        print(f"Episode: {episode:4d}\t|\tEpsilon: {self.epsilon:.4f}\t|\tMemory: {len(self.memory):5d}")
        print(f"Reward (avg {window}): {avg_reward:8.2f}\t|\tWin Rate: {win_rate:5.1f}%")
        print(f"Loss (avg {window}): {avg_loss:8.4f}\t|\tAvg Length: {avg_length:5.1f}")
        print(f"{'='*80}")
    
    # enregistre les metriques dans tensorboard avec temperature au lieu d'epsilon
    def log_to_tensorboard(self, episode: int, episode_reward: float, episode_loss: float, 
                        episode_length: int, win: bool = False, already_recorded: bool = False) -> None:
        
        # Enregistrer les statistiques actuelles si pas déjà fait
        if not already_recorded:
            self.record_episode_stats(episode_reward, episode_length)
            if episode_loss is not None:
                self.training_stats['losses'].append(episode_loss)
                if len(self.training_stats['losses']) > 1000:
                    self.training_stats['losses'].pop(0)
        
        with self.tensorboard_writer.as_default():
            tf.summary.scalar('Episode/Reward', episode_reward, step=episode)
            tf.summary.scalar('Episode/Loss', episode_loss, step=episode)
            tf.summary.scalar('Episode/Length', episode_length, step=episode)
            tf.summary.scalar('Episode/Epsilon', self.epsilon, step=episode)
            tf.summary.scalar('Episode/Memory_size', len(self.memory), step=episode)
            tf.summary.scalar('Episode/Win_rate', 1.0 if win else 0.0, step=episode)
            
            # moyenne des récompenses et pertes sur les 100 derniers épisodes
            if len(self.training_stats['rewards']) > 0:
                # 100 épisodes
                window_size = 100
                
                # avg_reward_100 = moyenne des récompenses sur les 100 derniers épisodes
                avg_reward_100 = np.mean(self.training_stats['rewards'][-window_size:])
                
                # avg_loss_100 = moyenne des pertes sur les 100 derniers épisodes
                avg_loss_100 = np.mean(self.training_stats['losses'][-window_size:]) if self.training_stats['losses'] else 0
                
                # nombre de wins / nombre d'épisodes 
                wins_in_window = len([r for r in self.training_stats['rewards'][-window_size:] if r > 0])
                win_rate_100 = wins_in_window / window_size
                
                tf.summary.scalar('Moving_average_100/Reward', avg_reward_100, step=episode)
                tf.summary.scalar('Moving_average_100/Loss', avg_loss_100, step=episode)
                tf.summary.scalar('Moving_average_100/Win_rate', win_rate_100, step=episode)

            self.tensorboard_writer.flush()

    # met à jour le modele cible avec les poids du modele principal
    def update_target_model(self) -> None:
        self.target_model.set_weights(self.q_model.get_weights())
    
    # sauvegarde et charge le modele
    def save_model(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.q_model.save_weights(filepath)
    
    # charge les poids du modele depuis un fichier
    def load_model(self, filepath: str) -> None:
        if os.path.exists(filepath):
            self.q_model.load_weights(filepath)
            self.target_model.set_weights(self.q_model.get_weights())
            print(f"modele chargé depuis {filepath}")
        else:
            print(f"fichier {filepath} non trouvé, utilisation d'un modele neuf")
    
    # affiche l'architecture du modele DQN
    def get_model_summary(self) -> None:
        print("\nArchitecture du modele DQN:")
        self.q_model.summary()
    
    # retourne la taille de la mémoire de replay
    def get_memory_size(self) -> int:
        return len(self.memory)
