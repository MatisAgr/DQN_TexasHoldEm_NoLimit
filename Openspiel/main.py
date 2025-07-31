import numpy as np
from keras import layers
import tensorflow as tf
import random
from collections import deque
import os
from enum import Enum
from typing import List, Tuple, Dict
from treys import Card as TreysCard, Evaluator, Deck
import time
import sys
from colorama import init, Fore, Back, Style

# Configuration des threads pour optimiser les performances
os.environ["OMP_NUM_THREADS"] = "14" 
tf.config.threading.set_intra_op_parallelism_threads(14)
tf.config.threading.set_inter_op_parallelism_threads(14)

# Initialisation de colorama pour les couleurs dans la console
init(autoreset=True)

# ========================= SIMULATION POKER AVEC TREYS =========================

class PokerAction(Enum):
    FOLD = 0
    CALL = 1
    RAISE_SMALL = 2
    RAISE_BIG = 3
    ALL_IN = 4

class PokerConsole:
    """Classe pour l'affichage console amélioré"""
    
    @staticmethod
    def clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def print_header(title):
        print(f"\n{Fore.CYAN}{'=' * 80}")
        print(f"{Fore.YELLOW}{title.center(80)}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    
    @staticmethod
    def print_cards(cards, label="Cartes"):
        cards_str = []
        for card in cards:
            card_str = TreysCard.int_to_pretty_str(card)
            # Colorier selon la couleur
            if '♥' in card_str or '♦' in card_str:
                cards_str.append(f"{Fore.RED}{card_str}{Style.RESET_ALL}")
            else:
                cards_str.append(f"{Fore.WHITE}{card_str}{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}{label}: {' '.join(cards_str)}{Style.RESET_ALL}")
    
    @staticmethod
    def print_game_state(env):
        print(f"\n{Fore.MAGENTA}┌─ ÉTAT DE LA PARTIE ─────────────────────────────────────────┐")
        print(f"│ Pot: {Fore.YELLOW}{env.pot:>6}{Style.RESET_ALL} chips     │     Round: {Fore.CYAN}{env.get_betting_round_name():<12}{Style.RESET_ALL} │")
        print(f"│ Mes jetons: {Fore.GREEN}{env.player_chips:>6}{Style.RESET_ALL}  │     Jetons Adv: {Fore.RED}{env.opponent_chips:>6}{Style.RESET_ALL}    │")
        print(f"│ Ma mise: {Fore.BLUE}{env.player_bet:>6}{Style.RESET_ALL}     │     Mise Adv: {Fore.BLUE}{env.opponent_bet:>6}{Style.RESET_ALL}      │")
        print(f"{Fore.MAGENTA}└─────────────────────────────────────────────────────────────┘{Style.RESET_ALL}")
    
    @staticmethod
    def print_action(player, action, amount=0):
        action_colors = {
            "FOLD": Fore.RED,
            "CALL": Fore.YELLOW,
            "RAISE_SMALL": Fore.CYAN,
            "RAISE_BIG": Fore.MAGENTA,
            "ALL_IN": Fore.RED + Style.BRIGHT
        }
        color = action_colors.get(action.name if hasattr(action, 'name') else str(action), Fore.WHITE)
        
        if amount > 0:
            print(f"{Fore.WHITE}{player}: {color}{action.name if hasattr(action, 'name') else action} ({amount} chips){Style.RESET_ALL}")
        else:
            print(f"{Fore.WHITE}{player}: {color}{action.name if hasattr(action, 'name') else action}{Style.RESET_ALL}")
    
    @staticmethod
    def print_hand_strength(strength, rank_class):
        strength_color = Fore.GREEN if strength > 4000 else Fore.YELLOW if strength > 2000 else Fore.RED
        print(f"Force de la main: {strength_color}{strength}{Style.RESET_ALL} ({rank_class})")

class TreysPokerEnv:
    def __init__(self):
        self.evaluator = Evaluator()
        self.num_actions = len(PokerAction)
        self.state_size = 15  # État optimisé pour treys
        self.console = PokerConsole()
        self.reset()
    
    def reset(self):
        """Initialise une nouvelle partie"""
        self.deck = Deck()
        self.deck.shuffle()
        
        # Distribution des cartes
        self.player_cards = [self.deck.draw(1)[0], self.deck.draw(1)[0]]
        self.opponent_cards = [self.deck.draw(1)[0], self.deck.draw(1)[0]]
        self.community_cards = []
        
        # État initial
        self.pot = 30  # Blinds
        self.player_chips = 1000
        self.opponent_chips = 1000
        self.betting_round = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        
        # Blinds alternées
        if random.random() < 0.5:
            self.player_bet = 10   # Small blind
            self.opponent_bet = 20  # Big blind
            self.player_position = "SB"
        else:
            self.player_bet = 20   # Big blind
            self.opponent_bet = 10  # Small blind
            self.player_position = "BB"
        
        self.done = False
        self.winner = None
        self.last_action = None
        self.hand_history = []
        
        return self._get_state()
    
    def get_betting_round_name(self):
        rounds = ["Pre-flop", "Flop", "Turn", "River", "Showdown"]
        return rounds[min(self.betting_round, 4)]
    
    def _get_hand_strength(self, hole_cards, community_cards=None):
        """Calcule la force de la main avec treys"""
        if community_cards is None:
            community_cards = self.community_cards
        
        if len(community_cards) < 3:
            # Pre-flop ou pas assez de cartes communes : évaluation simplifiée
            return self._get_preflop_strength(hole_cards)
        
        all_cards = hole_cards + community_cards
        if len(all_cards) < 5:
            # Pas assez de cartes pour une évaluation complète
            return self._get_preflop_strength(hole_cards) * 1000
        
        # Évaluation avec treys (plus le score est bas, meilleure est la main)
        hand_rank = self.evaluator.evaluate(all_cards[:5], all_cards[5:7] if len(all_cards) == 7 else [])
        
        # Conversion : treys donne un score où 1 = meilleur, 7462 = pire
        # On inverse pour que plus grand = meilleur
        normalized_strength = 7463 - hand_rank
        return normalized_strength
    
    def _get_hand_class(self, hole_cards, community_cards=None):
        """Retourne la classe de la main (paire, brelan, etc.)"""
        if community_cards is None:
            community_cards = self.community_cards
            
        if len(community_cards) < 3:
            return "Cartes hautes"
        
        all_cards = hole_cards + community_cards
        if len(all_cards) < 5:
            return "Cartes hautes"
        
        hand_rank = self.evaluator.evaluate(all_cards[:5], all_cards[5:7] if len(all_cards) == 7 else [])
        return self.evaluator.class_to_string(self.evaluator.get_rank_class(hand_rank))
    
    def _get_preflop_strength(self, cards):
        """Évalue la force pré-flop des cartes privées avec treys"""
        if len(cards) != 2:
            return 0
            
        # Conversion pour analyse
        card1_rank = TreysCard.get_rank_int(cards[0])
        card2_rank = TreysCard.get_rank_int(cards[1])
        card1_suit = TreysCard.get_suit_int(cards[0])
        card2_suit = TreysCard.get_suit_int(cards[1])
        
        high_rank = max(card1_rank, card2_rank)
        low_rank = min(card1_rank, card2_rank)
        is_suited = card1_suit == card2_suit
        is_pair = card1_rank == card2_rank
        gap = high_rank - low_rank
        
        if is_pair:
            if high_rank >= 10:  # Paires hautes (JJ+)
                return 0.9
            elif high_rank >= 6:   # Paires moyennes 
                return 0.7
            else:                  # Petites paires
                return 0.5
        
        if is_suited:
            if high_rank >= 12 or (high_rank >= 10 and gap <= 3):  # AKs, AQs, KQs, etc.
                return 0.8
            elif high_rank >= 8 and gap <= 4:  # Connecteurs suited moyens
                return 0.6
            else:
                return 0.3
        else:
            if high_rank >= 12 and low_rank >= 10:  # AK, AQ, KQ
                return 0.7
            elif high_rank >= 10 and gap <= 2:      # AJ, KJ, etc.
                return 0.5
            elif gap <= 1 and high_rank >= 7:       # Connecteurs
                return 0.4
            else:
                return 0.2
    
    def _get_state(self):
        """Retourne l'état actuel sous forme de vecteur optimisé"""
        # Force de la main
        hand_strength = self._get_hand_strength(self.player_cards)
        normalized_strength = hand_strength / 7463.0 if hand_strength > 1 else hand_strength
        
        # Informations sur les cartes privées
        card1_rank = TreysCard.get_rank_int(self.player_cards[0]) / 12.0
        card2_rank = TreysCard.get_rank_int(self.player_cards[1]) / 12.0
        is_suited = float(TreysCard.get_suit_int(self.player_cards[0]) == TreysCard.get_suit_int(self.player_cards[1]))
        is_pair = float(TreysCard.get_rank_int(self.player_cards[0]) == TreysCard.get_rank_int(self.player_cards[1]))
        
        state = [
            normalized_strength,                        # Force de la main normalisée
            self.pot / 2000.0,                         # Taille du pot
            self.player_chips / 2000.0,                # Mes jetons
            self.opponent_chips / 2000.0,              # Jetons adversaire
            self.betting_round / 3.0,                  # Round de mise
            len(self.community_cards) / 5.0,           # Nombre de cartes communes
            self.player_bet / 200.0,                   # Ma mise actuelle
            self.opponent_bet / 200.0,                 # Mise adversaire
            card1_rank,                                # Rang carte 1
            card2_rank,                                # Rang carte 2
            is_suited,                                 # Suited?
            is_pair,                                   # Paire?
            float(self.player_position == "BB"),       # Position (big blind?)
            abs(self.player_bet - self.opponent_bet) / 200.0,  # Différence de mise
            min(self.player_chips, self.opponent_chips) / 2000.0  # Stack le plus petit
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _deal_community_cards(self):
        """Distribue les cartes communes selon le round"""
        if self.betting_round == 1 and len(self.community_cards) == 0:  # Flop
            self.community_cards.extend(self.deck.draw(3))
        elif self.betting_round == 2 and len(self.community_cards) == 3:  # Turn
            self.community_cards.extend(self.deck.draw(1))
        elif self.betting_round == 3 and len(self.community_cards) == 4:  # River
            self.community_cards.extend(self.deck.draw(1))
    
    def _opponent_action(self):
        """Stratégie adversaire améliorée avec treys"""
        opp_strength = self._get_hand_strength(self.opponent_cards)
        preflop_strength = self._get_preflop_strength(self.opponent_cards)
        
        # Normalisation de la force
        if len(self.community_cards) >= 3:
            strength_factor = opp_strength / 7463.0
        else:
            strength_factor = preflop_strength
        
        # Facteurs de décision
        pot_odds = self.pot / max(50, abs(self.player_bet - self.opponent_bet) + 1)
        aggressive_factor = 1.2 if self.betting_round > 0 else 1.0
        
        # Décision basée sur la force réelle de la main
        if strength_factor > 0.7:  # Main très forte
            actions = [PokerAction.RAISE_BIG, PokerAction.RAISE_SMALL, PokerAction.CALL]
            weights = [0.5, 0.3, 0.2]
        elif strength_factor > 0.4:  # Main correcte
            actions = [PokerAction.CALL, PokerAction.RAISE_SMALL, PokerAction.FOLD]
            weights = [0.5, 0.3, 0.2]
        elif strength_factor > 0.2:  # Main marginale
            actions = [PokerAction.CALL, PokerAction.FOLD]
            weights = [0.3, 0.7]
        else:  # Main faible
            # Bluff occasionnel
            if random.random() < 0.1:
                return PokerAction.RAISE_SMALL
            actions = [PokerAction.FOLD, PokerAction.CALL]
            weights = [0.85, 0.15]
        
        return random.choices(actions, weights=weights)[0]
    
    def step(self, action_idx):
        if self.done:
            return self._get_state(), 0, True, {}
        
        action = PokerAction(action_idx)
        reward = 0
        self.last_action = action
        
        # Action du joueur
        if action == PokerAction.FOLD:
            self.done = True
            reward = -self.player_bet
            self.winner = "opponent"
            
        elif action == PokerAction.CALL:
            call_amount = max(0, self.opponent_bet - self.player_bet)
            if call_amount <= self.player_chips:
                self.player_chips -= call_amount
                self.player_bet += call_amount
                self.pot += call_amount
            
        elif action == PokerAction.RAISE_SMALL:
            raise_amount = max(50, self.opponent_bet - self.player_bet + 50)
            if raise_amount <= self.player_chips:
                self.player_chips -= raise_amount
                self.player_bet += raise_amount
                self.pot += raise_amount
            
        elif action == PokerAction.RAISE_BIG:
            raise_amount = max(100, self.opponent_bet - self.player_bet + 100)
            if raise_amount <= self.player_chips:
                self.player_chips -= raise_amount
                self.player_bet += raise_amount
                self.pot += raise_amount
            
        elif action == PokerAction.ALL_IN:
            all_in_amount = self.player_chips
            self.player_bet += all_in_amount
            self.pot += all_in_amount
            self.player_chips = 0
        
        # Action de l'adversaire (si le joueur n'a pas fold)
        if not self.done:
            opp_action = self._opponent_action()
            
            if opp_action == PokerAction.FOLD:
                self.done = True
                reward = self.pot - self.player_bet
                self.winner = "player"
                
            elif opp_action == PokerAction.CALL:
                call_amount = max(0, self.player_bet - self.opponent_bet)
                if call_amount <= self.opponent_chips:
                    self.opponent_chips -= call_amount
                    self.opponent_bet += call_amount
                    self.pot += call_amount
                    
            elif opp_action == PokerAction.RAISE_SMALL:
                raise_amount = max(50, self.player_bet - self.opponent_bet + 50)
                if raise_amount <= self.opponent_chips:
                    self.opponent_chips -= raise_amount
                    self.opponent_bet += raise_amount
                    self.pot += raise_amount
                    
            elif opp_action == PokerAction.RAISE_BIG:
                raise_amount = max(100, self.player_bet - self.opponent_bet + 100)
                if raise_amount <= self.opponent_chips:
                    self.opponent_chips -= raise_amount
                    self.opponent_bet += raise_amount
                    self.pot += raise_amount
            
            # Progression vers le prochain round
            if not self.done and self.player_bet == self.opponent_bet:
                self.betting_round += 1
                self._deal_community_cards()
                
                if self.betting_round > 3:  # Showdown
                    self.done = True
                    player_strength = self._get_hand_strength(self.player_cards)
                    opponent_strength = self._get_hand_strength(self.opponent_cards)
                    
                    if player_strength > opponent_strength:
                        reward = self.pot - self.player_bet
                        self.winner = "player"
                    elif player_strength < opponent_strength:
                        reward = -self.player_bet
                        self.winner = "opponent"  
                    else:  # Égalité
                        reward = 0
                        self.winner = "tie"
        
        return self._get_state(), reward, self.done, {"winner": self.winner}
    
    def render(self, show_opponent_cards=False):
        """Affichage console amélioré"""
        self.console.print_header(f"🎰 POKER DQN - {self.get_betting_round_name()}")
        
        # Cartes du joueur
        self.console.print_cards(self.player_cards, "🤖 Vos cartes")
        
        # Force de la main du joueur
        player_strength = self._get_hand_strength(self.player_cards)
        player_class = self._get_hand_class(self.player_cards)
        self.console.print_hand_strength(player_strength, player_class)
        
        # Cartes de l'adversaire (optionnel)
        if show_opponent_cards:
            self.console.print_cards(self.opponent_cards, "🎲 Cartes adversaire")
            opp_strength = self._get_hand_strength(self.opponent_cards)
            opp_class = self._get_hand_class(self.opponent_cards)
            print(f"{Fore.RED}Force adversaire: {opp_strength} ({opp_class}){Style.RESET_ALL}")
        
        # Cartes communes
        if self.community_cards:
            self.console.print_cards(self.community_cards, "🃏 Board")
        
        # État du jeu
        self.console.print_game_state(self)
        
        # Dernière action
        if self.last_action:
            self.console.print_action("🤖 Vous", self.last_action)
        
        # Résultat final
        if self.done:
            if self.winner == "player":
                print(f"\n{Fore.GREEN}🎉 VICTOIRE! Vous gagnez {self.pot - self.player_bet} chips!{Style.RESET_ALL}")
            elif self.winner == "opponent":
                print(f"\n{Fore.RED}💀 DÉFAITE! Vous perdez {self.player_bet} chips.{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}🤝 ÉGALITÉ! Personne ne gagne.{Style.RESET_ALL}")
            
            if show_opponent_cards:
                print(f"\n{Fore.CYAN}=== SHOWDOWN ==={Style.RESET_ALL}")
                self.console.print_cards(self.opponent_cards, "🎲 Cartes adversaire révélées")
        
        print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}")

class Card:
    def __init__(self, suit: int, rank: int):
        self.suit = suit  # 0-3 (♠♥♦♣)
        self.rank = rank  # 0-12 (2-A)
    
    def __str__(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['♠', '♥', '♦', '♣']
        return f"{ranks[self.rank]}{suits[self.suit]}"
    
    def value(self):
        return self.rank

# ========================= CONFIGURATION =========================

# Paramètres DQN  
epsilon = 1.0
epsilon_min = 0.05  # Plus d'exploration
epsilon_decay = 0.9995  # Décroissance plus lente
gamma = 0.95
batch_size = 32
memory_size = 100000
episodes = 3000  # Nombre d'épisodes d'entraînement
target_update_frequency = 50  # Mise à jour plus fréquente

# Paramètres d'entraînement
learning_rate = 0.001
train_frequency = 4

# Affichage
show_game_every = 250  # Afficher une partie toutes les N parties
show_opponent_cards = False  # Afficher les cartes de l'adversaire pendant l'entraînement

# ========================= INITIALISATION =========================

env = TreysPokerEnv()
state_size = env.state_size
num_actions = env.num_actions

print(f"{Fore.CYAN}🎰 Environnement Poker initialisé avec Treys")
print(f"📊 Dimension de l'état: {state_size}")
print(f"🎯 Nombre d'actions: {num_actions}")
print(f"🎮 Actions disponibles: {[action.name for action in PokerAction]}{Style.RESET_ALL}")

# ========================= CALLBACKS TENSORFLOW =========================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=15,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='../checkpoints/dqn_simple_poker.weights.h5',
        monitor='loss',
        save_best_only=True,
        save_weights_only=True
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='../logs',
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='batch'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.2,
        patience=8,
        min_lr=1e-6,
        verbose=1
    )
]

# ========================= MODÈLE DQN =========================

def create_q_model():
    """Crée le modèle Q-Network pour le poker"""
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(state_size,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model

# Création des modèles
q_model = create_q_model()
target_model = create_q_model()
target_model.set_weights(q_model.get_weights())

print("\n🧠 Architecture du modèle DQN:")
q_model.summary()

# ========================= MÉMOIRE DE REPLAY =========================

memory = deque(maxlen=memory_size)

def store_transition(state, action, reward, next_state, done):
    """Stocke une transition dans la mémoire de replay"""
    memory.append((state, action, reward, next_state, done))

def sample_batch():
    """Échantillonne un batch de transitions de la mémoire"""
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
    return states, actions, rewards, next_states, dones

# ========================= POLITIQUES =========================

def get_legal_actions():
    """Retourne toutes les actions légales (pour cette version simple, toutes le sont)"""
    return list(range(num_actions))

def epsilon_greedy_policy(state, epsilon):
    """Politique epsilon-greedy pour le poker"""
    if np.random.random() < epsilon:
        return np.random.choice(get_legal_actions())
    else:
        q_values = q_model.predict(state[np.newaxis], verbose=0)[0]
        return np.argmax(q_values)

# ========================= ENTRAÎNEMENT =========================

def train_step():
    """Effectue un pas d'entraînement du modèle DQN"""
    if len(memory) < batch_size:
        return
    
    states, actions, rewards, next_states, dones = sample_batch()
    
    # Calcul des Q-values cibles
    next_q_values = target_model.predict(next_states, verbose=0)
    max_next_q_values = np.max(next_q_values, axis=1)
    
    target_q_values = q_model.predict(states, verbose=0)
    
    for i in range(batch_size):
        if dones[i]:
            target_q_values[i][actions[i]] = rewards[i]
        else:
            target_q_values[i][actions[i]] = rewards[i] + gamma * max_next_q_values[i]
    
    # Entraîner le modèle
    history = q_model.fit(states, target_q_values, verbose=0, epochs=1)
    return history.history['loss'][0]

# ========================= BOUCLE PRINCIPALE D'ENTRAÎNEMENT =========================

reward_history = []
win_history = []
loss_history = []
step_count = 0

# Affichage initial
PokerConsole.clear_screen()
PokerConsole.print_header("🚀 ENTRAÎNEMENT DQN POKER AVEC TREYS")
print(f"{Fore.GREEN}🎯 Objectif: Entraîner un agent DQN pour jouer au poker Texas Hold'em")
print(f"📈 Épisodes prévus: {episodes}")
print(f"🧠 Architecture: Réseau de neurones avec TensorFlow/Keras")
print(f"🃏 Évaluation: Bibliothèque Treys pour évaluation précise des mains{Style.RESET_ALL}")
print("\n" + "🔄 Début de l'entraînement..." + "\n")

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    steps_in_episode = 0
    episode_loss = 0
    
    # Affichage de la partie si c'est le bon intervalle
    show_this_game = (episode % show_game_every == 0 and episode > 0)
    
    if show_this_game:
        PokerConsole.clear_screen()
        print(f"{Fore.YELLOW}🎮 Démonstration - Épisode {episode}{Style.RESET_ALL}")
        env.render(show_opponent_cards=show_opponent_cards)
        time.sleep(1)
    
    while True:
        # Action du joueur avec epsilon-greedy
        action = epsilon_greedy_policy(state, epsilon)
        
        next_state, reward, done, info = env.step(action)
        
        # Affichage de l'action si démonstration
        if show_this_game:
            action_enum = PokerAction(action)
            PokerConsole.print_action("🤖 Agent IA", action_enum)
            time.sleep(0.8)
            
            # Affichage de l'état après l'action
            env.render(show_opponent_cards=show_opponent_cards)
            if not done:
                time.sleep(1)
        
        # Stockage de la transition
        store_transition(state, action, reward, next_state, done)
        
        total_reward += reward
        state = next_state
        steps_in_episode += 1
        step_count += 1
        
        # Entraînement périodique
        if step_count % train_frequency == 0 and len(memory) >= batch_size:
            loss = train_step()
            if loss is not None:
                episode_loss += loss
        
        if done:
            if show_this_game:
                print(f"\n{Fore.CYAN}🏁 Fin de la partie - Récompense totale: {total_reward:.1f}{Style.RESET_ALL}")
                time.sleep(2)
            break
    
    # Mise à jour de l'epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Mise à jour du target network
    if episode % target_update_frequency == 0:
        target_model.set_weights(q_model.get_weights())
    
    # Statistiques
    reward_history.append(total_reward)
    win_history.append(1 if total_reward > 0 else 0)
    loss_history.append(episode_loss / max(1, steps_in_episode))
    
    # Affichage des résultats périodique
    if episode % 100 == 0:
        recent_wins = sum(win_history[-100:]) if len(win_history) >= 100 else sum(win_history)
        recent_episodes = min(100, len(win_history))
        win_rate = recent_wins / recent_episodes * 100
        avg_reward = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
        avg_loss = np.mean(loss_history[-100:]) if len(loss_history) >= 100 else np.mean(loss_history)
        
        # Affichage coloré des statistiques
        win_color = Fore.GREEN if win_rate > 60 else Fore.YELLOW if win_rate > 40 else Fore.RED
        reward_color = Fore.GREEN if avg_reward > 0 else Fore.RED
        
        print(f"{Fore.CYAN}🎲 Épisode {episode:4d}{Style.RESET_ALL} | "
              f"💰 Récompense: {reward_color}{total_reward:6.1f}{Style.RESET_ALL} | "
              f"🎯 Epsilon: {Fore.BLUE}{epsilon:.3f}{Style.RESET_ALL} | "
              f"🏆 Victoires: {win_color}{win_rate:5.1f}%{Style.RESET_ALL} | "
              f"📊 R.moy: {reward_color}{avg_reward:6.1f}{Style.RESET_ALL} | "
              f"📉 Loss: {Fore.MAGENTA}{avg_loss:.4f}{Style.RESET_ALL}")
    
    # Sauvegarde périodique
    if episode % 500 == 0 and episode > 0:
        q_model.save_weights(f'../checkpoints/dqn_treys_poker_episode_{episode}.weights.h5')
        print(f"{Fore.GREEN}💾 Modèle sauvegardé à l'épisode {episode}{Style.RESET_ALL}")

# ========================= RÉSULTATS FINAUX =========================

print("\n" + "=" * 70)
print("🏆 ENTRAÎNEMENT TERMINÉ!")
print("=" * 70)

final_win_rate = sum(win_history[-200:]) / min(200, len(win_history)) * 100
final_avg_reward = np.mean(reward_history[-200:])
final_avg_loss = np.mean(loss_history[-200:]) if loss_history else 0

print(f"{Fore.GREEN}📊 Statistiques finales:")
print(f"   • Épisodes totaux: {Fore.CYAN}{episodes}{Style.RESET_ALL}")
print(f"   • Taux de victoire final (200 derniers): {Fore.YELLOW}{final_win_rate:.1f}%{Style.RESET_ALL}")
print(f"   • Récompense moyenne finale: {Fore.BLUE}{final_avg_reward:.2f}{Style.RESET_ALL}")
print(f"   • Loss moyenne finale: {Fore.MAGENTA}{final_avg_loss:.4f}{Style.RESET_ALL}")
print(f"   • Mémoire de replay: {Fore.RED}{len(memory)}{Style.RESET_ALL} transitions")

# Sauvegarde finale
q_model.save_weights('../checkpoints/dqn_treys_poker_final.weights.h5')
print(f"{Fore.GREEN}💾 Modèle final sauvegardé!{Style.RESET_ALL}")

# ========================= TEST DU MODÈLE =========================

PokerConsole.print_header("🎯 TEST DU MODÈLE ENTRAÎNÉ")
print(f"\n{Fore.YELLOW}🧪 Test du modèle sur 20 parties (mode exploitation pur)...{Style.RESET_ALL}")
test_wins = 0
test_rewards = []

for test_episode in range(20):
    state = env.reset()
    episode_reward = 0
    
    # Affichage pour les 3 premières parties de test
    show_test = test_episode < 3
    
    if show_test:
        print(f"\n{Fore.CYAN}🎮 Partie de test #{test_episode + 1}{Style.RESET_ALL}")
        env.render(show_opponent_cards=True)
        time.sleep(1)
    
    while True:
        # Mode test : epsilon = 0 (pas d'exploration)
        q_values = q_model.predict(state[np.newaxis], verbose=0)[0]
        action = np.argmax(q_values)
        
        if show_test:
            action_enum = PokerAction(action)
            PokerConsole.print_action("🤖 Agent IA", action_enum)
            print(f"Q-values: {[f'{q:.2f}' for q in q_values]}")
            time.sleep(1)
        
        state, reward, done, info = env.step(action)
        episode_reward += reward
        
        if show_test:
            env.render(show_opponent_cards=True)
            if not done:
                time.sleep(1)
        
        if done:
            if show_test:
                print(f"\n{Fore.CYAN}🏁 Fin de la partie de test - Récompense: {episode_reward:.1f}{Style.RESET_ALL}")
                time.sleep(2)
            break
    
    test_rewards.append(episode_reward)
    if episode_reward > 0:
        test_wins += 1
    
    # Affichage résumé pour chaque test
    result_color = Fore.GREEN if episode_reward > 0 else Fore.RED
    result_emoji = "✅" if episode_reward > 0 else "❌"
    
    print(f"Test {test_episode + 1:2d}: {result_color}Récompense = {episode_reward:6.1f}{Style.RESET_ALL} "
          f"{result_emoji} {'Victoire' if episode_reward > 0 else 'Défaite'}")

test_win_rate = test_wins / 20 * 100

# Résultats finaux du test
PokerConsole.print_header("🎯 RÉSULTATS DU TEST FINAL")
print(f"{Fore.GREEN}🏆 Victoires: {Fore.YELLOW}{test_wins}/20{Style.RESET_ALL} ({Fore.CYAN}{test_win_rate:.1f}%{Style.RESET_ALL})")
print(f"{Fore.BLUE}📊 Récompense moyenne: {Fore.YELLOW}{np.mean(test_rewards):.2f}{Style.RESET_ALL}")
print(f"{Fore.GREEN}🚀 Meilleure récompense: {Fore.YELLOW}{max(test_rewards):.1f}{Style.RESET_ALL}")
print(f"{Fore.RED}📉 Pire récompense: {Fore.YELLOW}{min(test_rewards):.1f}{Style.RESET_ALL}")

# Message final
if test_win_rate > 60:
    print(f"\n{Fore.GREEN}� Excellent! Votre agent DQN performe très bien au poker!{Style.RESET_ALL}")
elif test_win_rate > 40:
    print(f"\n{Fore.YELLOW}🎯 Bon travail! Votre agent DQN a des performances correctes.{Style.RESET_ALL}")
else:
    print(f"\n{Fore.RED}🔧 L'agent a encore du mal. Peut-être faut-il plus d'entraînement?{Style.RESET_ALL}")

print(f"\n{Fore.CYAN}🎰 Votre agent DQN pour le poker avec Treys est prêt!")
print(f"💡 L'évaluation des mains est maintenant précise grâce à la bibliothèque Treys.{Style.RESET_ALL}")
print(f"{Fore.MAGENTA}🚀 Améliorations possibles: bluff plus sophistiqué, position, stack management...{Style.RESET_ALL}")