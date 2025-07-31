import numpy as np
from keras import layers
import tensorflow as tf
import random
from collections import deque
import os
from enum import Enum
from typing import List, Tuple, Dict

# Configuration des threads pour optimiser les performances
os.environ["OMP_NUM_THREADS"] = "14" 
tf.config.threading.set_intra_op_parallelism_threads(14)
tf.config.threading.set_inter_op_parallelism_threads(14)

# ========================= SIMULATION POKER SIMPLIFIÉE =========================

class PokerAction(Enum):
    FOLD = 0
    CALL = 1
    RAISE_SMALL = 2
    RAISE_BIG = 3
    ALL_IN = 4

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

class SimplePokerEnv:
    def __init__(self):
        self.num_actions = len(PokerAction)
        self.state_size = 15  # [hand_strength, pot_size, my_chips, opp_chips, betting_round, position, ...]
        self.reset()
    
    def reset(self):
        self.deck = self._create_deck()
        self.player_cards = [self._draw_card(), self._draw_card()]
        self.opponent_cards = [self._draw_card(), self._draw_card()]
        self.community_cards = []
        self.pot = 30  # Blinds augmentées
        self.player_chips = 1000
        self.opponent_chips = 1000
        self.betting_round = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        # Alternance des positions (qui paye la big blind)
        if random.random() < 0.5:
            self.player_bet = 10   # Small blind
            self.opponent_bet = 20  # Big blind
        else:
            self.player_bet = 20   # Big blind
            self.opponent_bet = 10  # Small blind
        self.done = False
        self.winner = None
        
        return self._get_state()
    
    def _create_deck(self):
        return [Card(suit, rank) for suit in range(4) for rank in range(13)]
    
    def _draw_card(self):
        return self.deck.pop(random.randint(0, len(self.deck) - 1))
    
    def _get_hand_strength(self, cards):
        """Calcule la force de la main (améliorée)"""
        if len(cards) < 2:
            return 0
        
        values = sorted([card.value() for card in cards], reverse=True)
        suits = [card.suit for card in cards]
        
        # Détection des combinaisons
        value_counts = {}
        for v in values:
            value_counts[v] = value_counts.get(v, 0) + 1
        
        # Flush (couleur)
        is_flush = len(cards) >= 5 and len(set(suits)) == 1
        
        # Straight (suite) - simplifié
        is_straight = False
        if len(cards) >= 5:
            unique_values = sorted(set(values), reverse=True)
            if len(unique_values) >= 5:
                for i in range(len(unique_values) - 4):
                    if unique_values[i] - unique_values[i+4] == 4:
                        is_straight = True
                        break
        
        # Évaluation hiérarchique
        pairs = [v for v, count in value_counts.items() if count == 2]
        trips = [v for v, count in value_counts.items() if count == 3]
        quads = [v for v, count in value_counts.items() if count == 4]
        
        if is_straight and is_flush:
            return 0.9 + max(values) * 0.001  # Straight flush
        elif quads:
            return 0.8 + max(quads) * 0.001   # Carré
        elif trips and pairs:
            return 0.7 + max(trips) * 0.001   # Full house
        elif is_flush:
            return 0.6 + max(values) * 0.001  # Couleur
        elif is_straight:
            return 0.5 + max(values) * 0.001  # Suite
        elif trips:
            return 0.4 + max(trips) * 0.001   # Brelan
        elif len(pairs) >= 2:
            return 0.3 + max(pairs) * 0.001   # Double paire
        elif pairs:
            return 0.2 + max(pairs) * 0.001   # Paire
        else:
            return 0.05 + max(values) * 0.001  # Carte haute
    
    def _get_state(self):
        """Retourne l'état actuel sous forme de vecteur"""
        all_cards = self.player_cards + self.community_cards
        hand_strength = self._get_hand_strength(all_cards)
        
        state = [
            hand_strength,                              # Force de la main
            self.pot / 2000.0,                         # Taille du pot (normalisée)
            self.player_chips / 2000.0,                # Mes jetons (normalisés)
            self.opponent_chips / 2000.0,              # Jetons adversaire (normalisés)
            self.betting_round / 3.0,                  # Round de mise (normalisé)
            len(self.community_cards) / 5.0,           # Nombre de cartes communes
            self.player_bet / 200.0,                   # Ma mise actuelle
            self.opponent_bet / 200.0,                 # Mise adversaire
            max([card.value() for card in self.player_cards]) / 12.0,  # Carte haute
            min([card.value() for card in self.player_cards]) / 12.0,  # Carte basse
            int(self.player_cards[0].suit == self.player_cards[1].suit),  # Suited?
            abs(self.player_cards[0].value() - self.player_cards[1].value()) / 12.0,  # Gap
            random.random() * 0.1,  # Noise pour la variabilité
            random.random() * 0.1,  # Noise pour la variabilité
            random.random() * 0.1   # Noise pour la variabilité
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _deal_community_cards(self):
        """Distribue les cartes communes selon le round"""
        if self.betting_round == 1 and len(self.community_cards) == 0:  # Flop
            for _ in range(3):
                self.community_cards.append(self._draw_card())
        elif self.betting_round == 2 and len(self.community_cards) == 3:  # Turn
            self.community_cards.append(self._draw_card())
        elif self.betting_round == 3 and len(self.community_cards) == 4:  # River
            self.community_cards.append(self._draw_card())
    
    def _opponent_action(self):
        """Stratégie adversaire améliorée et plus agressive"""
        opp_cards = self.opponent_cards + self.community_cards
        opp_strength = self._get_hand_strength(opp_cards)
        
        # Évaluation du potentiel (cartes privées seulement)
        preflop_strength = self._get_preflop_strength(self.opponent_cards)
        
        # Facteurs de décision
        pot_odds = self.pot / max(50, self.player_bet - self.opponent_bet + 1)
        position_factor = 1.1 if self.betting_round > 0 else 1.0  # Plus agressif post-flop
        
        # Décision basée sur plusieurs facteurs
        if opp_strength > 0.6 or preflop_strength > 0.8:  # Main très forte
            actions = [PokerAction.RAISE_BIG, PokerAction.RAISE_SMALL, PokerAction.CALL]
            weights = [0.4, 0.4, 0.2]
        elif opp_strength > 0.3 or preflop_strength > 0.5:  # Main correcte
            actions = [PokerAction.CALL, PokerAction.RAISE_SMALL, PokerAction.FOLD]
            weights = [0.5, 0.3, 0.2]
        elif opp_strength > 0.15 or (preflop_strength > 0.3 and pot_odds > 2):  # Main marginale
            actions = [PokerAction.CALL, PokerAction.FOLD]
            weights = [0.4, 0.6]
        else:  # Main faible
            # Plus de bluff occasionnel
            if random.random() < 0.15:  # 15% de bluff
                return PokerAction.RAISE_SMALL
            actions = [PokerAction.FOLD, PokerAction.CALL]
            weights = [0.8, 0.2]
        
        return random.choices(actions, weights=weights)[0]
    
    def _get_preflop_strength(self, cards):
        """Évalue la force pré-flop des cartes privées"""
        if len(cards) != 2:
            return 0
            
        c1, c2 = cards[0], cards[1]
        high_card = max(c1.value(), c2.value())
        low_card = min(c1.value(), c2.value())
        is_suited = c1.suit == c2.suit
        is_pair = c1.value() == c2.value()
        gap = high_card - low_card
        
        if is_pair:
            if high_card >= 10:  # Paires hautes (JJ+)
                return 0.9
            elif high_card >= 7:   # Paires moyennes (77-TT)
                return 0.7
            else:                  # Petites paires
                return 0.5
        
        if is_suited:
            if high_card >= 12 or (high_card >= 10 and gap <= 3):  # AKs, AQs, KQs, etc.
                return 0.8
            elif high_card >= 9 and gap <= 4:  # Connecteurs suited moyens
                return 0.6
            else:
                return 0.3
        else:
            if high_card >= 12 and low_card >= 10:  # AK, AQ, KQ
                return 0.7
            elif high_card >= 11 and gap <= 2:      # AJ, KJ, etc.
                return 0.5
            elif gap <= 1 and high_card >= 8:       # Connecteurs
                return 0.4
            else:
                return 0.2
    
    def step(self, action_idx):
        if self.done:
            return self._get_state(), 0, True, {}
        
        action = PokerAction(action_idx)
        reward = 0
        
        # Action du joueur
        if action == PokerAction.FOLD:
            self.done = True
            reward = -self.player_bet  # Perte de la mise
            self.winner = "opponent"
        
        elif action == PokerAction.CALL:
            call_amount = max(0, self.opponent_bet - self.player_bet)
            if call_amount <= self.player_chips:
                self.player_chips -= call_amount
                self.player_bet += call_amount
                self.pot += call_amount
        
        elif action == PokerAction.RAISE_SMALL:
            raise_amount = max(50, self.opponent_bet - self.player_bet + 50)  # Raise minimum
            if raise_amount <= self.player_chips:
                self.player_chips -= raise_amount
                self.player_bet += raise_amount
                self.pot += raise_amount
        
        elif action == PokerAction.RAISE_BIG:
            raise_amount = max(100, self.opponent_bet - self.player_bet + 100)  # Raise plus gros
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
                reward = self.pot - self.player_bet  # Gain du pot
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
            if not self.done:
                if self.player_bet == self.opponent_bet:
                    self.betting_round += 1
                    self._deal_community_cards()
                    
                    if self.betting_round > 3:  # Showdown
                        self.done = True
                        player_strength = self._get_hand_strength(self.player_cards + self.community_cards)
                        opponent_strength = self._get_hand_strength(self.opponent_cards + self.community_cards)
                        
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
    
    def render(self):
        print(f"\n=== POKER GAME ===")
        print(f"Cartes joueur: {[str(card) for card in self.player_cards]}")
        print(f"Cartes communes: {[str(card) for card in self.community_cards]}")
        print(f"Pot: {self.pot} | Mes jetons: {self.player_chips} | Adv jetons: {self.opponent_chips}")
        print(f"Round: {self.betting_round} | Ma mise: {self.player_bet} | Mise adv: {self.opponent_bet}")
        if self.done:
            print(f"Partie terminée - Gagnant: {self.winner}")

# ========================= CONFIGURATION DQN =========================

# Paramètres DQN  
epsilon = 1.0
epsilon_min = 0.05  # Plus d'exploration
epsilon_decay = 0.9995  # Décroissance plus lente
gamma = 0.95
batch_size = 32
memory_size = 100000
episodes = 5000  # Plus d'épisodes
target_update_frequency = 50  # Mise à jour plus fréquente

# Paramètres d'entraînement
learning_rate = 0.001
train_frequency = 4

# ========================= INITIALISATION =========================

env = SimplePokerEnv()
state_size = env.state_size
num_actions = env.num_actions

print(f"🎰 Environnement Poker simplifié initialisé")
print(f"Dimension de l'état: {state_size}")
print(f"Nombre d'actions: {num_actions}")
print(f"Actions disponibles: {[action.name for action in PokerAction]}")

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

print(f"\n🚀 Début de l'entraînement DQN pour le Poker")
print("=" * 70)

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    steps_in_episode = 0
    episode_loss = 0
    
    while True:
        # Action du joueur avec epsilon-greedy
        action = epsilon_greedy_policy(state, epsilon)
        
        next_state, reward, done, info = env.step(action)
        
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
    
    # Affichage des résultats
    if episode % 100 == 0:
        recent_wins = sum(win_history[-100:]) if len(win_history) >= 100 else sum(win_history)
        recent_episodes = min(100, len(win_history))
        win_rate = recent_wins / recent_episodes * 100
        avg_reward = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
        avg_loss = np.mean(loss_history[-100:]) if len(loss_history) >= 100 else np.mean(loss_history)
        
        print(f"🎲 Épisode {episode:4d} | "
              f"Récompense: {total_reward:6.1f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Victoires: {win_rate:5.1f}% | "
              f"R.moy: {avg_reward:6.1f} | "
              f"Loss: {avg_loss:.4f}")
    
    # Démonstration occasionnelle
    if episode % 500 == 0 and episode > 0:
        print(f"\n🎮 Démonstration de la partie #{episode}:")
        env.render()
        
        # Sauvegarde
        q_model.save_weights(f'../checkpoints/dqn_poker_episode_{episode}.weights.h5')
        print(f"💾 Modèle sauvegardé à l'épisode {episode}")

# ========================= RÉSULTATS FINAUX =========================

print("\n" + "=" * 70)
print("🏆 ENTRAÎNEMENT TERMINÉ!")
print("=" * 70)

final_win_rate = sum(win_history[-200:]) / min(200, len(win_history)) * 100
final_avg_reward = np.mean(reward_history[-200:])
final_avg_loss = np.mean(loss_history[-200:]) if loss_history else 0

print(f"📊 Statistiques finales:")
print(f"   • Épisodes totaux: {episodes}")
print(f"   • Taux de victoire final (200 derniers): {final_win_rate:.1f}%")
print(f"   • Récompense moyenne finale: {final_avg_reward:.2f}")
print(f"   • Loss moyenne finale: {final_avg_loss:.4f}")
print(f"   • Mémoire de replay: {len(memory)} transitions")

# Sauvegarde finale
q_model.save_weights('../checkpoints/dqn_simple_poker_final.weights.h5')
print("💾 Modèle final sauvegardé!")

# ========================= TEST DU MODÈLE =========================

print(f"\n� Test du modèle entraîné (20 parties)...")
test_wins = 0
test_rewards = []

for test_episode in range(20):
    state = env.reset()
    episode_reward = 0
    
    while True:
        # Mode test : epsilon = 0 (pas d'exploration)
        q_values = q_model.predict(state[np.newaxis], verbose=0)[0]
        action = np.argmax(q_values)
        
        state, reward, done, info = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    test_rewards.append(episode_reward)
    if episode_reward > 0:
        test_wins += 1
    
    print(f"Test {test_episode + 1:2d}: Récompense = {episode_reward:6.1f} "
          f"({'✅ Victoire' if episode_reward > 0 else '❌ Défaite'})")

test_win_rate = test_wins / 20 * 100
print(f"\n🎯 Résultats du test final:")
print(f"   • Victoires: {test_wins}/20 ({test_win_rate:.1f}%)")
print(f"   • Récompense moyenne: {np.mean(test_rewards):.2f}")
print(f"   • Meilleure récompense: {max(test_rewards):.1f}")
print(f"   • Pire récompense: {min(test_rewards):.1f}")

print(f"\n🎰 Votre agent DQN pour le poker est prêt !")
print("💡 Améliorations possibles: règles plus complexes, évaluation de mains réelle, bluff...")