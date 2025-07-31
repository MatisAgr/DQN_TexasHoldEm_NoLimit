# config

# ========================= PARAMÈTRES DQN =========================

class DQNConfig:
    """Configuration pour l'agent DQN"""
    
    # Paramètres d'apprentissage
    LEARNING_RATE = 0.001
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.9995
    GAMMA = 0.95  # Facteur de discount
    
    # Mémoire de replay
    MEMORY_SIZE = 100000
    BATCH_SIZE = 32
    
    # Architecture du réseau
    HIDDEN_LAYERS = [128, 128, 64, 32]
    DROPOUT_RATE = 0.3


class TrainingConfig:
    """Configuration pour l'entraînement"""
    
    # Nombre d'épisodes et fréquences
    EPISODES = 3000
    TRAIN_FREQUENCY = 4  # Entraîner tous les N steps
    TARGET_UPDATE_FREQUENCY = 50  # Mettre à jour le modèle cible tous les N épisodes
    
    # Affichage et sauvegarde
    SHOW_GAME_EVERY = 1  # Afficher une partie tous les N épisodes
    SAVE_EVERY = 500  # Sauvegarder tous les N épisodes
    PROGRESS_EVERY = 100  # Afficher les statistiques tous les N épisodes
    
    # Options d'affichage
    SHOW_OPPONENT_CARDS = True  # Afficher les cartes adversaires pendant l'entraînement


class PokerConfig:
    """Configuration pour l'environnement de poker"""
    
    # Configuration initiale
    INITIAL_CHIPS = 1000
    SMALL_BLIND = 10
    BIG_BLIND = 20
    
    # Actions et mises
    SMALL_RAISE = 50
    BIG_RAISE = 100
    
    # État du jeu
    STATE_SIZE = 15
    NUM_ACTIONS = 5  # FOLD, CALL, RAISE_SMALL, RAISE_BIG, ALL_IN


class PathConfig:
    """Configuration des chemins de fichiers"""
    
    # Répertoires
    CHECKPOINTS_DIR = "checkpoints"
    LOGS_DIR = "logs"
    
    # Fichiers de sauvegarde
    FINAL_MODEL = "checkpoints/dqn_treys_poker_final.weights.h5"
    EPISODE_MODEL_TEMPLATE = "checkpoints/dqn_treys_poker_episode_{}.weights.h5"
    
    # Logs TensorBoard
    TENSORBOARD_LOG_DIR = "logs"


# ========================= CONFIGURATION GLOBALE =========================

class Config:
    """Configuration globale du projet"""
    
    DQN = DQNConfig()
    TRAINING = TrainingConfig()
    POKER = PokerConfig()
    PATHS = PathConfig()
    
    # Informations sur le projet
    PROJECT_NAME = "DQN Texas Hold'em No Limit"
    VERSION = "1.0.0"
    
    # Configuration système
    TENSORFLOW_THREADS = 14
    RANDOM_SEED = 42


# Fonction pour charger une configuration personnalisée
def load_config_from_file(config_file: str = None) -> Config:
    """
    Charge une configuration depuis un fichier (future amélioration)
    
    Args:
        config_file: Chemin vers le fichier de configuration
        
    Returns:
        Instance de Config
    """
    # Pour l'instant, retourne la configuration par défaut
    # Dans le futur, on pourrait implémenter le chargement depuis JSON/YAML
    return Config()


# Configuration par défaut
config = Config()
