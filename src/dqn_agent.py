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
                 temperature_start: float = None, temperature_min: float = None, temperature_decay: float = None,
                 gamma: float = None, memory_size: int = None):
        

        self.state_size = state_size
        self.num_actions = num_actions
        
        # Paramètres d'apprentissage
        self.learning_rate = learning_rate if learning_rate is not None else config.DQN.LEARNING_RATE
        self.temperature = temperature_start if temperature_start is not None else 2.0
        self.temperature_min = temperature_min if temperature_min is not None else 0.1
        self.temperature_decay = temperature_decay if temperature_decay is not None else 0.995
        self.gamma = gamma if gamma is not None else config.DQN.GAMMA
        self.batch_size = config.DQN.BATCH_SIZE
        
        # Mémoire de replay
        memory_size = memory_size if memory_size is not None else config.DQN.MEMORY_SIZE
        self.memory = deque(maxlen=memory_size)
        
        # Statistiques d'entraînement
        self.training_stats = {
            'losses': [],
            'rewards': [],
            'episode_lengths': [],
            'q_values': [],
            'temperature_history': []
        }
        
        # TensorBoard writer personnalisé
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_log_dir = os.path.join(config.PATHS.TENSORBOARD_LOG_DIR, f"dqn_training_{timestamp}")
        self.tensorboard_writer = tf.summary.create_file_writer(self.tensorboard_log_dir)
        
        # copie des poids
        self.q_model = self._create_model()
        self.target_model = self._create_model()
        self.target_model.set_weights(self.q_model.get_weights())
    
    # creation du modele
    def _create_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential()
        
        # Couche d'entrée
        model.add(layers.Dense(config.DQN.HIDDEN_LAYERS[0], activation='relu', input_shape=(self.state_size,)))
        model.add(layers.Dropout(config.DQN.DROPOUT_RATE))
        
        # Couches cachées selon la configuration
        for i, layer_size in enumerate(config.DQN.HIDDEN_LAYERS[1:], 1):
            model.add(layers.Dense(layer_size, activation='relu'))
            # Moins de dropout dans les dernières couches
            dropout_rate = config.DQN.DROPOUT_RATE * (0.7 if i >= len(config.DQN.HIDDEN_LAYERS) - 1 else 1.0)
            model.add(layers.Dropout(dropout_rate))
        
        # Couche de sortie
        model.add(layers.Dense(self.num_actions, activation='linear'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model
    
    # callbacks
    def get_callbacks(self, episode: int = 0) -> List[tf.keras.callbacks.Callback]:
        # Créer un dossier unique pour cette session d'entrainement
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_log_dir = os.path.join(config.PATHS.TENSORBOARD_LOG_DIR, f"training_{timestamp}")
        
        return [
            # TensorBoard pour voir les courbes d'entrainement
            tf.keras.callbacks.TensorBoard(
                log_dir=tensorboard_log_dir,
                histogram_freq=0,  # Pas d'histogrammes pour éviter les problèmes
                write_graph=False,  # Pas de graphe pour éviter les problèmes
                write_images=False,  # Pas d'images pour éviter les problèmes
                update_freq='epoch',  # Mise à jour par époque plutôt que par batch
                profile_batch=0  # Pas de profiling pour éviter les problèmes
            ),
            # Sauvegarde le meilleur modele
            tf.keras.callbacks.ModelCheckpoint(
                filepath=config.PATHS.FINAL_MODEL,
                monitor='loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=0  # Pas de messages de sauvegarde
            ),
            # Reduit le learning rate si pas d'amelioration
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=20,
                min_lr=1e-7,
                verbose=0  # Pas de messages
            )
        ]
    
    # stocke une transition dans la memoire de replay
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))
    
    # retourne toutes les actions legales (elles le sont toutes mdr)
    def get_legal_actions(self) -> List[int]:
        return list(range(self.num_actions))
    
    # choisit une action selon e-greedy
    def act(self, state: np.ndarray, training: bool = True, temperature: float = 1.0) -> int:
        """choisit une action avec softmax au lieu d'epsilon-greedy"""
        q_values = self.q_model.predict(state[np.newaxis], verbose=0)[0]
        
        if training and temperature > 0:
            # softmax avec temperature pour l'exploration
            exp_q = np.exp(q_values / temperature)
            probabilities = exp_q / np.sum(exp_q)
            return np.random.choice(len(q_values), p=probabilities)
        else:
            # choix deterministe
            return np.argmax(q_values)


    def update_temperature(self, decay_rate: float = None) -> None:
        """reduit la temperature pour moins explorer"""
        decay = decay_rate if decay_rate is not None else self.temperature_decay
        self.temperature = max(self.temperature_min, self.temperature * decay)
        
        # enregistrer l'historique de temperature
        self.training_stats['temperature_history'].append(self.temperature)
        if len(self.training_stats['temperature_history']) > 1000:
            self.training_stats['temperature_history'].pop(0)
    
    # retourne les Q-values pour un etat donné
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return self.q_model.predict(state[np.newaxis], verbose=0)[0]
    
    # echantillonne un batch de transitions de la mémoire
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    # effectue un pas d'entraînement du modele DQN
    def train_step(self, batch_size: int = None, use_callbacks: bool = True) -> Optional[float]:
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        
        # Calcul des Q-values cibles
        next_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1) # Q(s', a') pour les états suivants
        
        target_q_values = self.q_model.predict(states, verbose=0) # Q(s, a) pour les etats actuels
        
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i] # Q(s, a) = r si l'etat suivant est terminal
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * max_next_q_values[i] # Q(s, a) = r + gamma * max_a' Q(s', a')
        
        # Entraîner le modèle avec ou sans callbacks
        if use_callbacks:
            callbacks = self.get_callbacks(episode=0)  # Nous n'avons pas le numéro d'épisode ici
            history = self.q_model.fit(states, target_q_values, verbose=0, epochs=1, callbacks=callbacks)
        else:
            history = self.q_model.fit(states, target_q_values, verbose=0, epochs=1)
        
        loss = history.history['loss'][0]
        
        # Enregistrer les statistiques
        self.training_stats['losses'].append(loss) # enregistrer la perte
        if len(self.training_stats['losses']) > 1000:  # garder seulement les 1000 dernières
            self.training_stats['losses'].pop(0) # garder la mémoire légère
            
        return loss
    
    # Met à jour l'epsilon pour la politique e-greedy
    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # reduction exponentielle de l'epsilon 
        
        # Enregistrer l'historique d'epsilon
        self.training_stats['epsilon_history'].append(self.epsilon)
        if len(self.training_stats['epsilon_history']) > 1000:
            self.training_stats['epsilon_history'].pop(0)
    
    # enregistre les statistiques d'un épisode
    # episode_reward: récompense totale de l'épisode, episode_length: longueur de l'épisode, avg_q_value: q-value moyenne de l'épisode
    def record_episode_stats(self, episode_reward: float, episode_length: int, avg_q_value: float = None) -> None:
        """
        Enregistre les statistiques d'un épisode
        
        Args:
            episode_reward: Récompense totale de l'épisode
            episode_length: Nombre de steps dans l'épisode
            avg_q_value: Q-value moyenne de l'épisode (optionnel)
        """
        self.training_stats['rewards'].append(episode_reward)
        self.training_stats['episode_lengths'].append(episode_length)
        
        if avg_q_value is not None:
            self.training_stats['q_values'].append(avg_q_value)
        
        # Garder seulement les 1000 dernières entrées
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
        print(f"episode: {episode:4d} | temperature: {self.temperature:.4f} | memory: {len(self.memory):5d}")
        print(f"reward (avg {window}): {avg_reward:8.2f} | win rate: {win_rate:5.1f}%")
        print(f"loss (avg {window}):   {avg_loss:8.4f} | avg length: {avg_length:5.1f}")
        print(f"{'='*80}")
    
    def log_to_tensorboard(self, episode: int, episode_reward: float, episode_loss: float, 
                          episode_length: int, win: bool = False) -> None:
        """
        enregistre les metriques dans tensorboard avec temperature au lieu d'epsilon
        """
        with self.tensorboard_writer.as_default():
            tf.summary.scalar('episode/reward', episode_reward, step=episode)
            tf.summary.scalar('episode/loss', episode_loss, step=episode)
            tf.summary.scalar('episode/length', episode_length, step=episode)
            tf.summary.scalar('episode/temperature', self.temperature, step=episode)  # remplacer epsilon
            tf.summary.scalar('episode/memory_size', len(self.memory), step=episode)
            tf.summary.scalar('episode/win_rate', 1.0 if win else 0.0, step=episode)
            
            # moyennes mobiles sur 100 episodes
            if len(self.training_stats['rewards']) >= 100:
                avg_reward_100 = np.mean(self.training_stats['rewards'][-100:])
                avg_loss_100 = np.mean(self.training_stats['losses'][-100:]) if self.training_stats['losses'] else 0
                win_rate_100 = len([r for r in self.training_stats['rewards'][-100:] if r > 0]) / 100
                
                tf.summary.scalar('moving_average_100/reward', avg_reward_100, step=episode)
                tf.summary.scalar('moving_average_100/loss', avg_loss_100, step=episode)
                tf.summary.scalar('moving_average_100/win_rate', win_rate_100, step=episode)
            
            self.tensorboard_writer.flush()
    
    def close_tensorboard(self) -> None:
        """Ferme le writer TensorBoard"""
        if hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.close()
    
    def update_target_model(self) -> None:
        """Met à jour le modèle cible avec les poids du modèle principal"""
        self.target_model.set_weights(self.q_model.get_weights())
    
    def save_model(self, filepath: str) -> None:
        """Sauvegarde les poids du modèle"""
        # Créer le répertoire s'il n'existe pas
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.q_model.save_weights(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Charge les poids du modèle"""
        if os.path.exists(filepath):
            self.q_model.load_weights(filepath)
            self.target_model.set_weights(self.q_model.get_weights())
            print(f"Modèle chargé depuis {filepath}")
        else:
            print(f"Fichier {filepath} non trouvé, utilisation d'un modèle neuf")
    
    def get_model_summary(self) -> None:
        """Affiche l'architecture du reseau de neurones"""
        print("\nArchitecture du modele DQN:")
        self.q_model.summary()
    
    def get_memory_size(self) -> int:
        """Retourne la taille actuelle de la mémoire de replay"""
        return len(self.memory)
