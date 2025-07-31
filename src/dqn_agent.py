"""
Module contenant l'agent DQN (Deep Q-Network) pour le poker.
"""

import numpy as np
import tensorflow as tf
from keras import layers
import random
from collections import deque
from typing import List, Tuple, Optional
import os
from config import config


class DQNAgent:
    """Agent DQN pour le poker Texas Hold'em"""
    
    def __init__(self, state_size: int, num_actions: int, learning_rate: float = None,
                 epsilon: float = None, epsilon_min: float = None, epsilon_decay: float = None,
                 gamma: float = None, memory_size: int = None):
        """
        Initialise l'agent DQN
        
        Args:
            state_size: Dimension de l'espace d'état
            num_actions: Nombre d'actions possibles
            learning_rate: Taux d'apprentissage (utilise la config si None)
            epsilon: Probabilité d'exploration initiale (utilise la config si None)
            epsilon_min: Probabilité d'exploration minimale (utilise la config si None)
            epsilon_decay: Facteur de décroissance d'epsilon (utilise la config si None)
            gamma: Facteur de discount (utilise la config si None)
            memory_size: Taille de la mémoire de replay (utilise la config si None)
        """
        self.state_size = state_size
        self.num_actions = num_actions
        
        # Utilisation des paramètres de configuration par défaut si non spécifiés
        self.learning_rate = learning_rate if learning_rate is not None else config.DQN.LEARNING_RATE
        self.epsilon = epsilon if epsilon is not None else config.DQN.EPSILON_START
        self.epsilon_min = epsilon_min if epsilon_min is not None else config.DQN.EPSILON_MIN
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else config.DQN.EPSILON_DECAY
        self.gamma = gamma if gamma is not None else config.DQN.GAMMA
        self.batch_size = config.DQN.BATCH_SIZE
        
        # Mémoire de replay
        memory_size = memory_size if memory_size is not None else config.DQN.MEMORY_SIZE
        self.memory = deque(maxlen=memory_size)
        
        # Réseaux de neurones
        self.q_model = self._create_model()
        self.target_model = self._create_model()
        self.target_model.set_weights(self.q_model.get_weights())
    
    def _create_model(self) -> tf.keras.Model:
        """Crée le modèle Q-Network pour le poker"""
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
    
    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Callbacks pour l'entrainement
        TensorBoard: pour voir les graphiques d'entrainement
        Usage: ouvrir un terminal et taper 'tensorboard --logdir=logs'
        puis aller sur http://localhost:6006 dans le navigateur
        """
        return [
            # Arrete l'entrainement si pas d'amelioration
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=15,
                restore_best_weights=True
            ),
            # Sauvegarde le meilleur modele
            tf.keras.callbacks.ModelCheckpoint(
                filepath=config.PATHS.FINAL_MODEL,
                monitor='loss',
                save_best_only=True,
                save_weights_only=True
            ),
            # TensorBoard pour voir les courbes d'entrainement
            tf.keras.callbacks.TensorBoard(
                log_dir=config.PATHS.TENSORBOARD_LOG_DIR,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='batch'
            ),
            # Reduit le learning rate si pas d'amelioration
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.2,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool) -> None:
        """Stocke une transition dans la mémoire de replay"""
        self.memory.append((state, action, reward, next_state, done))
    
    def get_legal_actions(self) -> List[int]:
        """Retourne toutes les actions légales (pour cette version simple, toutes le sont)"""
        return list(range(self.num_actions))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choisit une action selon la politique epsilon-greedy
        
        Args:
            state: État actuel
            training: Si True, utilise epsilon-greedy, sinon exploitation pure
            
        Returns:
            Action à exécuter
        """
        if training and np.random.random() < self.epsilon:
            return np.random.choice(self.get_legal_actions())
        else:
            q_values = self.q_model.predict(state[np.newaxis], verbose=0)[0]
            return np.argmax(q_values)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Retourne les Q-values pour un état donné"""
        return self.q_model.predict(state[np.newaxis], verbose=0)[0]
    
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Échantillonne un batch de transitions de la mémoire"""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def train_step(self, batch_size: int = None) -> Optional[float]:
        """
        Effectue un pas d'entraînement du modèle DQN
        
        Args:
            batch_size: Taille du batch d'entraînement (utilise la config si None)
            
        Returns:
            Loss de l'entraînement ou None si pas assez de données
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        
        # Calcul des Q-values cibles
        next_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1)
        
        target_q_values = self.q_model.predict(states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * max_next_q_values[i]
        
        # Entraîner le modèle
        history = self.q_model.fit(states, target_q_values, verbose=0, epochs=1)
        return history.history['loss'][0]
    
    def update_epsilon(self) -> None:
        """Met à jour epsilon selon la stratégie de décroissance"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
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
