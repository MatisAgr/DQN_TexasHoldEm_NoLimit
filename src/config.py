# config

# ========================= PARAMÈTRES DQN =========================

# config agent DQN
class DQNConfig:
    
    # apprentissage
    LEARNING_RATE = 0.001   # taux d'apprentissage
    EPSILON_START = 1.0     # epsilon initial pour l'exploration
    EPSILON_MIN = 0.05      # epsilon minimum pour l'exploration
    EPSILON_DECAY = 0.9995  # taux de décroissance de l'exploration
    GAMMA = 0.95            # facteur d'actualisation (importance des récompenses futures (long terme))
    
    TEMPERATURE_START = 2.0   # température initiale pour l'exploration softmax
    TEMPERATURE_MIN = 0.1     # température minimum pour l'exploration softmax
    TEMPERATURE_DECAY = 0.995 # taux de décroissance de la température
    
    # mémoire
    MEMORY_SIZE = 100000
    BATCH_SIZE = 32
    
    # réseau de neurones
    HIDDEN_LAYERS = [128, 128, 64, 32]
    DROPOUT_RATE = 0.3

# env training
class TrainingConfig: 
    
    # Nombre d'épisodes et fréquences
    EPISODES = 10000
    TRAIN_FREQUENCY = 4 
    TARGET_UPDATE_FREQUENCY = 50    # 
    
    # Affichage et sauvegarde
    SHOW_GAME_EVERY = 2000              # afficher une partie tous les n épisodes (avec l'affichage stylé dans le terminal)
    SAVE_EVERY = 2000                   # save tous les n épisodes
    PROGRESS_EVERY = 100                # résumé stats tous les n épisodes
    
    SHOW_OPPONENT_CARDS = True          # Afficher les cartes adversaires pendant l'entraînement
    
    # TODO: Mode IA vs IA (pas prêt)
    # MULTI_AGENT_MODE = False          # faire du IA vs IA ou IA vs bot
    # AGENT_NAMES = ["IA-1", "IA-2"]    # nom des deux agents
    # ALTERNATE_TRAINING = True         # alterner l'entraînement des deux agents
    # SHARED_EXPERIENCE = False         # partager la mémoire de replay entre agents (truc de fou mais je vais oublier l'idée )

# env poker
class PokerConfig:
    
    INITIAL_CHIPS = 1000
    SMALL_BLIND = 10
    BIG_BLIND = 20
    
    # mises
    SMALL_RAISE = 50
    BIG_RAISE = 100
    
    STATE_SIZE = 15 # vecteur d'état de l'environnement
    NUM_ACTIONS = 5  # 0: FOLD, 1: CALL, 2: RAISE_SMALL, 3: RAISE_BIG, 4: ALL_IN


class PathConfig:
    """Configuration des chemins de fichiers"""
    
    # paths pour les données
    CHECKPOINTS_DIR = "checkpoints"
    LOGS_DIR = "logs"
    TENSORBOARD_LOG_DIR = "logs"
    
    # path pour les modèles
    FINAL_MODEL = "checkpoints/dqn_treys_poker_final.weights.h5"
    EPISODE_MODEL_TEMPLATE = "checkpoints/dqn_treys_poker_episode_{}.weights.h5"
    
    # path pour les modèles multi-agent (pas encore implémenté)
    # AGENT1_FINAL_MODEL = "checkpoints/agent1_dqn_final.weights.h5"
    # AGENT2_FINAL_MODEL = "checkpoints/agent2_dqn_final.weights.h5"
    # AGENT1_EPISODE_TEMPLATE = "checkpoints/agent1_episode_{}.weights.h5"
    # AGENT2_EPISODE_TEMPLATE = "checkpoints/agent2_episode_{}.weights.h5"
    

# ========================= CONFIGURATION GLOBALE =========================

# global config
class Config:
    
    DQN = DQNConfig()
    TRAINING = TrainingConfig()
    POKER = PokerConfig()
    PATHS = PathConfig()
    
    # specifique à votre pc
    # TENSORFLOW_THREADS = 14   #n'a pas l'air de marcher
    RANDOM_SEED = 42            # le sens de la vie

config = Config()
