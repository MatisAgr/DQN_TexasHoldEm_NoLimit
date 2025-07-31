# logique du jeu de poker Texas Hold'em

import numpy as np
import random
from enum import Enum
from typing import List, Tuple, Dict, Optional
from treys import Card as TreysCard, Evaluator, Deck
from config import config


class PokerAction(Enum):
    """Actions possibles au poker"""
    FOLD = 0
    CALL = 1
    RAISE_SMALL = 2
    RAISE_BIG = 3
    ALL_IN = 4


class TreysPokerEnv:
    """Environnement de poker Texas Hold'em utilisant la bibliothèque Treys"""
    
    def __init__(self):
        self.evaluator = Evaluator()
        self.num_actions = config.POKER.NUM_ACTIONS
        self.state_size = config.POKER.STATE_SIZE
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Initialise une nouvelle partie"""
        self.deck = Deck()
        self.deck.shuffle()
        
        # Distribution des cartes
        self.player_cards = [self.deck.draw(1)[0], self.deck.draw(1)[0]]
        self.opponent_cards = [self.deck.draw(1)[0], self.deck.draw(1)[0]]
        self.community_cards = []
        
        # État initial
        self.pot = config.POKER.SMALL_BLIND + config.POKER.BIG_BLIND  # Blinds
        self.player_chips = config.POKER.INITIAL_CHIPS
        self.opponent_chips = config.POKER.INITIAL_CHIPS
        self.betting_round = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        
        # Blinds alternées
        if random.random() < 0.5:
            self.player_bet = config.POKER.SMALL_BLIND   # Small blind
            self.opponent_bet = config.POKER.BIG_BLIND  # Big blind
            self.player_position = "SB"
        else:
            self.player_bet = config.POKER.BIG_BLIND   # Big blind
            self.opponent_bet = config.POKER.SMALL_BLIND  # Small blind
            self.player_position = "BB"
        
        self.done = False
        self.winner = None
        self.last_action = None
        self.opponent_last_action = None
        self.hand_history = []
        
        return self._get_state()
    
    def get_betting_round_name(self) -> str:
        """Retourne le nom du round de mise actuel"""
        rounds = ["Pre-flop", "Flop", "Turn", "River", "Showdown"]
        return rounds[min(self.betting_round, 4)]
    
    def _get_hand_strength(self, hole_cards: List[int], community_cards: Optional[List[int]] = None) -> float:
        """la lib trey calcul la force de la main et donne un score"""
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
    
    def _get_hand_class(self, hole_cards: List[int], community_cards: Optional[List[int]] = None) -> str:
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
    
    def _get_preflop_strength(self, cards: List[int]) -> float:
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
    
    def _get_state(self) -> np.ndarray:
        """Retourne l'état actuel sous forme de vecteur optimisé"""
        # Force de la main
        hand_strength = self._get_hand_strength(self.player_cards)
        normalized_strength = hand_strength / 7463.0 if hand_strength > 1 else hand_strength
        
        # Informations sur les cartes privées
        card1_rank = TreysCard.get_rank_int(self.player_cards[0]) / 12.0
        card2_rank = TreysCard.get_rank_int(self.player_cards[1]) / 12.0
        is_suited = float(TreysCard.get_suit_int(self.player_cards[0]) == TreysCard.get_suit_int(self.player_cards[1]))
        is_pair = float(TreysCard.get_rank_int(self.player_cards[0]) == TreysCard.get_rank_int(self.player_cards[1]))
        
        # Normalisation avec les paramètres de configuration
        max_chips = config.POKER.INITIAL_CHIPS * 2  # Référence pour la normalisation
        max_bet = config.POKER.INITIAL_CHIPS
        
        state = [
            normalized_strength,                                         # Force de la main normalisée
            self.pot / max_chips,                                        # Taille du pot
            self.player_chips / max_chips,                               # Mes jetons
            self.opponent_chips / max_chips,                             # Jetons adversaire
            self.betting_round / 3.0,                                    # Round de mise
            len(self.community_cards) / 5.0,                             # Nombre de cartes communes
            self.player_bet / max_bet,                                   # Ma mise actuelle
            self.opponent_bet / max_bet,                                 # Mise adversaire
            card1_rank,                                                  # Rang carte 1
            card2_rank,                                                  # Rang carte 2
            is_suited,                                                   # Suited?
            is_pair,                                                     # Paire?
            float(self.player_position == "BB"),                         # Position (if big blind?)
            abs(self.player_bet - self.opponent_bet) / max_bet,          # Différence de mise
            min(self.player_chips, self.opponent_chips) / max_chips      # Stack le plus petit
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _deal_community_cards(self) -> None:
        """Distribue les cartes communes selon le round"""
        if self.betting_round == 1 and len(self.community_cards) == 0:  # Flop
            self.community_cards.extend(self.deck.draw(3))
        elif self.betting_round == 2 and len(self.community_cards) == 3:  # Turn
            self.community_cards.extend(self.deck.draw(1))
        elif self.betting_round == 3 and len(self.community_cards) == 4:  # River
            self.community_cards.extend(self.deck.draw(1))
    
    def _opponent_action(self) -> PokerAction:
        """Stratégie adversaire améliorée avec treys"""
        opp_strength = self._get_hand_strength(self.opponent_cards)
        preflop_strength = self._get_preflop_strength(self.opponent_cards)
        
        # Normalisation de la force
        if len(self.community_cards) >= 3:
            strength_factor = opp_strength / 7463.0
        else:
            strength_factor = preflop_strength
        
        # Facteurs de décision
        pot_odds = self.pot / max(config.POKER.SMALL_RAISE, abs(self.player_bet - self.opponent_bet) + 1)
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
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Exécute une action et retourne le nouvel état"""
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
            raise_amount = max(config.POKER.SMALL_RAISE, self.opponent_bet - self.player_bet + config.POKER.SMALL_RAISE)
            if raise_amount <= self.player_chips:
                self.player_chips -= raise_amount
                self.player_bet += raise_amount
                self.pot += raise_amount
            
        elif action == PokerAction.RAISE_BIG:
            raise_amount = max(config.POKER.BIG_RAISE, self.opponent_bet - self.player_bet + config.POKER.BIG_RAISE)
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
            self.opponent_last_action = opp_action  # Stocker l'action adverse pour l'affichage
            
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
    
    def get_player_hand_info(self) -> Tuple[float, str]:
        """Retourne les informations sur la main du joueur"""
        strength = self._get_hand_strength(self.player_cards)
        hand_class = self._get_hand_class(self.player_cards)
        return strength, hand_class
    
    def get_opponent_hand_info(self) -> Tuple[float, str]:
        """Retourne les informations sur la main de l'adversaire"""
        strength = self._get_hand_strength(self.opponent_cards)
        hand_class = self._get_hand_class(self.opponent_cards)
        return strength, hand_class
