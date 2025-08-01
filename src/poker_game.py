# logique du jeu de poker Texas Hold'em

import numpy as np
import random
from enum import Enum
from typing import List, Tuple, Dict, Optional
from treys import Card as TreysCard, Evaluator, Deck
from config import config


# action possible
class PokerAction(Enum):
    FOLD = 0
    CHECK = 1           # checker (ne rien miser si pas d'aggression)
    CALL = 2            # suivre la mise de l'adversaire
    RAISE_SMALL = 3     # petite relance (50 jetons)
    RAISE_BIG = 4       # grosse relance (100 jetons)
    ALL_IN = 5          # tapis

# env de la game de poker
class TreysPokerEnv:
    
    def __init__(self):
        self.evaluator = Evaluator()
        self.num_actions = config.POKER.NUM_ACTIONS
        self.state_size = config.POKER.STATE_SIZE
        self.reset()
    
    # reset l'env
    def reset(self) -> np.ndarray:
        self.deck = Deck()
        self.deck.shuffle()
        
        # disibution des cartes
        self.player_cards = [self.deck.draw(1)[0], self.deck.draw(1)[0]]
        self.opponent_cards = [self.deck.draw(1)[0], self.deck.draw(1)[0]]
        self.community_cards = []
        
        # etat init
        self.pot = config.POKER.SMALL_BLIND + config.POKER.BIG_BLIND  # Blinds
        self.player_chips = config.POKER.INITIAL_CHIPS
        self.opponent_chips = config.POKER.INITIAL_CHIPS
        self.betting_round = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        
        # alterner les blinds
        if random.random() < 0.5:
            self.player_bet = config.POKER.SMALL_BLIND      # Small blind
            self.opponent_bet = config.POKER.BIG_BLIND      # Big blind
            self.player_position = "SB"
            self.opponent_position = "BB"
            self.current_turn = "player"  # SB joue en premier pre-flop
            self.player_chips -= config.POKER.SMALL_BLIND
            self.opponent_chips -= config.POKER.BIG_BLIND
        else:
            self.player_bet = config.POKER.BIG_BLIND        # Big blind
            self.opponent_bet = config.POKER.SMALL_BLIND    # Small blind
            self.player_position = "BB"
            self.opponent_position = "SB"
            self.current_turn = "opponent"  # SB (opponent) joue en premier pre-flop
            self.player_chips -= config.POKER.BIG_BLIND
            self.opponent_chips -= config.POKER.SMALL_BLIND
        
        self.done = False
        self.winner = None
        self.last_action = None
        self.opponent_last_action = None
        self.hand_history = []
        self.action_history = []  # Historique des actions pour l'affichage
        
        return self.get_state()
    
    # nom du round
    def get_betting_round_name(self) -> str:
        # Pré-flop = reception des cartes
        # Flop = ouverture des cartes communes (3 cartes)
        # Turn = 4ème carte commune
        # River = 5ème carte commune
        # Showdown = révélation des mains
        rounds = ["Pre-flop", "Flop", "Turn", "River", "Showdown"]
        return rounds[min(self.betting_round, 4)]
    
    # actions légales
    def get_legal_actions(self) -> List[int]:
        legal_actions = []
        
        # si l'IA n'a plus de jetons, elle ne peut plus agir (sauf si showdown)
        if self.player_chips == 0:
            return []  # aucune action possible si all-in
        
        # si bot n'a plus de jetons non plus, showdown
        if self.opponent_chips == 0 and self.player_chips == 0:
            return []  # Showdown automatique
        
        # actions normales disponibles
        legal_actions.append(0)  # FOLD toujours possible
        
        if self.player_bet == self.opponent_bet:
            legal_actions.append(1)  # CHECK possible si mises égales
        
        if self.opponent_bet > self.player_bet:
            legal_actions.append(2)  # CALL possible si adversaire a misé plus
        
        # RAISE seulement si on a assez de jetons
        if self.player_chips >= config.POKER.SMALL_RAISE:
            legal_actions.append(3)  # RAISE_SMALL
        
        if self.player_chips >= config.POKER.BIG_RAISE:
            legal_actions.append(4)  # RAISE_BIG
        
        if self.player_chips > 0:
            legal_actions.append(5)  # ALL_IN toujours possible si on a des jetons
        
        return legal_actions
    
    # la lib trey calcul la force de la main et donne un score
    def get_hand_strength(self, hole_cards: List[int], community_cards: Optional[List[int]] = None) -> float:
        
        if community_cards is None:
            community_cards = self.community_cards
        
        # si moins de 3 sur le board -> force pré-flop
        if len(community_cards) < 3:
            return self.get_preflop_strength(hole_cards)
        
        
        # all_cards = cartes privées + cartes communes
        all_cards = hole_cards + community_cards
        
        if len(all_cards) < 5:
            return self.get_preflop_strength(hole_cards) * 1000
        
        # eval avec treys (plus le score est bas, meilleure est la main)
        hand_rank = self.evaluator.evaluate(all_cards[:5], all_cards[5:7] if len(all_cards) == 7 else [])
        
        # calcul avec treys 1 = meilleur, 7462 = pire
        # on inverse pour que plus grand = meilleur (plus simple)
        normalized_strength = 7463 - hand_rank
        return normalized_strength
    
    # donne le nom de la main
    # ex: paire, brelan, quinte, flush, full ...
    def get_hand_class(self, hole_cards: List[int], community_cards: Optional[List[int]] = None) -> str:
        if community_cards is None:
            community_cards = self.community_cards
            
        if len(community_cards) < 3:
            return "High Card" # car treys ne peut pas évaluer moins de 5 cartes
        
        all_cards = hole_cards + community_cards
        if len(all_cards) < 5: 
            return "High Card" # car treys ne peut pas évaluer moins de 5 cartes
        
        hand_rank = self.evaluator.evaluate(all_cards[:5], all_cards[5:7] if len(all_cards) == 7 else [])
        return self.evaluator.class_to_string(self.evaluator.get_rank_class(hand_rank))
    
    # force de la main pré-flop
    def get_preflop_strength(self, cards: List[int]) -> float:
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
            if high_rank >= 10:     # Paires hautes (JJ+)
                return 0.9
            elif high_rank >= 6:    # Paires moyennes 
                return 0.7
            else:                   # Petites paires
                return 0.5
        
        if is_suited:
            if high_rank >= 12 or (high_rank >= 10 and gap <= 3):
                return 0.8
            elif high_rank >= 8 and gap <= 4:
                return 0.6
            else:
                return 0.3
        else:
            if high_rank >= 12 and low_rank >= 10:
                return 0.7
            elif high_rank >= 10 and gap <= 2:
                return 0.5
            elif gap <= 1 and high_rank >= 7:
                return 0.4
            else:
                return 0.2
    
    # retourne l'état actuel sous forme de vecteur optimisé
    def get_state(self) -> np.ndarray:
        # force de la main
        hand_strength = self.get_hand_strength(self.player_cards)
        normalized_strength = hand_strength / 7463.0 if hand_strength > 1 else hand_strength
        
        # info sur les cartes privées
        card1_rank = TreysCard.get_rank_int(self.player_cards[0]) / 12.0
        card2_rank = TreysCard.get_rank_int(self.player_cards[1]) / 12.0
        is_suited = float(TreysCard.get_suit_int(self.player_cards[0]) == TreysCard.get_suit_int(self.player_cards[1]))
        is_pair = float(TreysCard.get_rank_int(self.player_cards[0]) == TreysCard.get_rank_int(self.player_cards[1]))
        
        # normalisation avec les paramètres de configuration
        max_chips = config.POKER.INITIAL_CHIPS * 2  # référence pour la normalisation
        max_bet = config.POKER.INITIAL_CHIPS
        
        # vecteur d'état pour l'agent DQN
        state = [
            normalized_strength,                                         # Force de la main normalisée
            self.pot / max_chips,                                        # Taille du pot
            self.player_chips / max_chips,                               # Jetons IA
            self.opponent_chips / max_chips,                             # Jetons adversaire
            self.betting_round / 3.0,                                    # Round de mise
            len(self.community_cards) / 5.0,                             # Nombre de cartes communes
            self.player_bet / max_bet,                                   # IA mise actuelle
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
    
    # distribue les cartes sur le board
    def deal_community_cards(self) -> None:
        if self.betting_round == 1 and len(self.community_cards) == 0:  # Flop
            self.community_cards.extend(self.deck.draw(3))
        elif self.betting_round == 2 and len(self.community_cards) == 3:  # Turn
            self.community_cards.extend(self.deck.draw(1))
        elif self.betting_round == 3 and len(self.community_cards) == 4:  # River
            self.community_cards.extend(self.deck.draw(1))
    
    # stratégie du bot
    def opponent_action(self) -> PokerAction:
        opp_strength = self.get_hand_strength(self.opponent_cards)
        preflop_strength = self.get_preflop_strength(self.opponent_cards)
        
        # normalisation de la force
        if len(self.community_cards) >= 3:
            strength_factor = opp_strength / 7463.0
        else:
            strength_factor = preflop_strength
        
        # facteur d'agressivité
        random_factor = random.random()
        
        # stratégie
        if strength_factor > 0.7:  # Main très forte
            actions = [PokerAction.RAISE_BIG, PokerAction.RAISE_SMALL, PokerAction.ALL_IN, PokerAction.CALL, PokerAction.CHECK]
            weights = [0.3, 0.25, 0.15, 0.15, 0.15]
        elif strength_factor > 0.4:  # Main correcte
            actions = [PokerAction.CALL, PokerAction.RAISE_SMALL, PokerAction.CHECK, PokerAction.RAISE_BIG, PokerAction.FOLD]
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        elif strength_factor > 0.2:  # Main marginale
            actions = [PokerAction.CHECK, PokerAction.CALL, PokerAction.RAISE_SMALL, PokerAction.FOLD]
            weights = [0.4, 0.3, 0.2, 0.1]
        else:  # Main faible
            if random_factor < 0.3:  # 30% de bluff
                actions = [PokerAction.RAISE_SMALL, PokerAction.RAISE_BIG]
                weights = [0.7, 0.3]
                return random.choices(actions, weights=weights)[0]
            else:
                actions = [PokerAction.CHECK, PokerAction.FOLD, PokerAction.CALL]
                weights = [0.5, 0.3, 0.2]
        
        return random.choices(actions, weights=weights)[0]
    
    # exécute une action et retourne le nouvel état
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            return self.get_state(), 0, True, {}
        
        # si l'IA n'a plus de jetons, elle ne peut plus agir
        if self.player_chips == 0:
            # passer directement au showdown ou à la suite
            return self.handle_all_in_situation()
        
        action = PokerAction(action_idx)
        reward = 0
        self.last_action = action
        
        # enregistrer l'action du joueur dans l'historique
        action_name = action.name
        action_amount = 0
        
        # calculer le montant de l'action pour l'historique
        if action == PokerAction.CHECK:
            action_amount = 0
        elif action == PokerAction.CALL:
            action_amount = max(0, self.opponent_bet - self.player_bet)
        elif action == PokerAction.RAISE_SMALL:
            action_amount = max(0, self.opponent_bet - self.player_bet) + config.POKER.SMALL_RAISE
        elif action == PokerAction.RAISE_BIG:
            action_amount = max(0, self.opponent_bet - self.player_bet) + config.POKER.BIG_RAISE
        elif action == PokerAction.ALL_IN:
            action_amount = self.player_chips
        
        # ajouter à l'historique seulement si l'action a un montant > 0 ou n'est pas ALL_IN redondant
        if action != PokerAction.ALL_IN or action_amount > 0:
            self.action_history.append(("IA", action_name, action_amount))
        
        # Changer le tour après l'action
        self.current_turn = "opponent"
        
        # action du joueur
        if action == PokerAction.FOLD:
            self.done = True
            reward = -self.player_bet
            self.winner = "opponent"
            
        elif action == PokerAction.CHECK:
            # CHECK: ne rien faire si les mises sont égales
            # Si l'adversaire a misé plus, CHECK n'est pas valide, on traite comme un CALL
            if self.opponent_bet > self.player_bet:
                call_amount = self.opponent_bet - self.player_bet
                if call_amount <= self.player_chips:
                    self.player_chips -= call_amount
                    self.player_bet += call_amount
                    self.pot += call_amount
            # Sinon CHECK: aucune action nécessaire
            
        elif action == PokerAction.CALL:
            # calculer le montant nécessaire pour égaliser
            call_amount = max(0, self.opponent_bet - self.player_bet)
            if call_amount <= self.player_chips:
                self.player_chips -= call_amount
                self.player_bet += call_amount
                self.pot += call_amount
            
        elif action == PokerAction.RAISE_SMALL:
            # calculer la relance totale nécessaire
            call_amount = max(0, self.opponent_bet - self.player_bet)
            total_raise = call_amount + config.POKER.SMALL_RAISE
            if total_raise <= self.player_chips:
                self.player_chips -= total_raise
                self.player_bet += total_raise
                self.pot += total_raise
            
        elif action == PokerAction.RAISE_BIG:
            # calculer la grosse relance totale nécessaire
            call_amount = max(0, self.opponent_bet - self.player_bet)
            total_raise = call_amount + config.POKER.BIG_RAISE
            if total_raise <= self.player_chips:
                self.player_chips -= total_raise
                self.player_bet += total_raise
                self.pot += total_raise
            
        elif action == PokerAction.ALL_IN:
            all_in_amount = self.player_chips
            self.player_bet += all_in_amount
            self.pot += all_in_amount
            self.player_chips = 0
        
        # action de l'adversaire (si IA n'a pas fold)
        if not self.done:
            opp_action = self.opponent_action()
            self.opponent_last_action = opp_action  # stocker l'action adverse pour l'affichage
            
            # Enregistrer l'action de l'adversaire
            opp_action_name = opp_action.name
            opp_action_amount = 0
            
            if opp_action == PokerAction.FOLD:
                self.done = True
                # Gain = pot total moins ce qu'on a investi
                reward = self.pot - self.player_bet
                self.winner = "player"
                self.action_history.append(("Adversaire", opp_action_name, 0))
                
            elif opp_action == PokerAction.CHECK:
                # CHECK: ne rien faire si les mises sont égales
                # Si le joueur a misé plus, CHECK n'est pas valide, on traite comme un CALL
                if self.player_bet > self.opponent_bet:
                    call_amount = self.player_bet - self.opponent_bet
                    if call_amount <= self.opponent_chips:
                        self.opponent_chips -= call_amount
                        self.opponent_bet += call_amount
                        self.pot += call_amount
                        opp_action_amount = call_amount
                self.action_history.append(("Adversaire", opp_action_name, opp_action_amount))
                
            elif opp_action == PokerAction.CALL:
                # adversaire égalise notre mise
                call_amount = max(0, self.player_bet - self.opponent_bet)
                if call_amount <= self.opponent_chips:
                    self.opponent_chips -= call_amount
                    self.opponent_bet += call_amount
                    self.pot += call_amount
                    opp_action_amount = call_amount
                self.action_history.append(("Adversaire", opp_action_name, opp_action_amount))
                    
            elif opp_action == PokerAction.RAISE_SMALL:
                # adversaire relance
                call_amount = max(0, self.player_bet - self.opponent_bet)
                total_raise = call_amount + config.POKER.SMALL_RAISE
                if total_raise <= self.opponent_chips:
                    self.opponent_chips -= total_raise
                    self.opponent_bet += total_raise
                    self.pot += total_raise
                    opp_action_amount = total_raise
                self.action_history.append(("Adversaire", opp_action_name, opp_action_amount))
                    
            elif opp_action == PokerAction.RAISE_BIG:
                # adversaire fait une grosse relance
                call_amount = max(0, self.player_bet - self.opponent_bet)
                total_raise = call_amount + config.POKER.BIG_RAISE
                if total_raise <= self.opponent_chips:
                    self.opponent_chips -= total_raise
                    self.opponent_bet += total_raise
                    self.pot += total_raise
                    opp_action_amount = total_raise
                self.action_history.append(("Adversaire", opp_action_name, opp_action_amount))
                    
            elif opp_action == PokerAction.ALL_IN:
                # adversaire fait tapis
                all_in_amount = self.opponent_chips
                self.opponent_bet += all_in_amount
                self.pot += all_in_amount
                self.opponent_chips = 0
                self.action_history.append(("Adversaire", opp_action_name, all_in_amount))
            
            # Remettre le tour au joueur après l'action de l'adversaire
            if not self.done:
                self.current_turn = "player"
            
            # next betting round si pas de fold
            if not self.done and self.player_bet == self.opponent_bet:
                self.betting_round += 1
                self.deal_community_cards()
                
                if self.betting_round > 3:  # Showdown
                    self.done = True
                    player_strength = self.get_hand_strength(self.player_cards)
                    opponent_strength = self.get_hand_strength(self.opponent_cards)
                    
                    if player_strength > opponent_strength:
                        reward = self.pot - self.player_bet
                        self.winner = "player"
                    elif player_strength < opponent_strength:
                        reward = -self.player_bet
                        self.winner = "opponent"  
                    else:  # egalité
                        reward = 0
                        self.winner = "tie"
        
        return self.get_state(), reward, self.done, {"winner": self.winner}
    
    # afficher la main de l'IA
    def get_player_hand_info(self) -> Tuple[float, str]:
        """Retourne les informations sur la main du joueur"""
        strength = self.get_hand_strength(self.player_cards)
        hand_class = self.get_hand_class(self.player_cards)
        return strength, hand_class

    # afficher la main de l'adversaire    
    def get_opponent_hand_info(self) -> Tuple[float, str]:
        strength = self.get_hand_strength(self.opponent_cards)
        hand_class = self.get_hand_class(self.opponent_cards)
        return strength, hand_class





    # TODO: revoir le ALL_IN (loop / crash / move illégal de temps en temps)
 
    # gérer la situation ALL_IN si ia plus de jetons
    def handle_all_in_situation(self) -> Tuple[np.ndarray, float, bool, Dict]:
        # si les deux joueurs sont ALL_IN, aller directement au showdown
        if self.player_chips == 0 and self.opponent_chips == 0:
            return self.proceed_to_showdown()
        
        # si seule l'IA est ALL_IN, l'adversaire doit répondre
        if self.player_chips == 0 and self.opponent_chips > 0:
            opp_action = self.opponent_action()
            
            # action de l'adversaire
            if opp_action == PokerAction.FOLD:
                self.done = True
                reward = self.pot - self.player_bet
                self.winner = "player"
                return self.get_state(), reward, True, {"winner": self.winner}
            
            elif opp_action == PokerAction.CALL:
                # adversaire égalise, les deux sont maintenant ALL_IN ou au showdown
                call_amount = min(self.opponent_chips, self.player_bet - self.opponent_bet)
                if call_amount > 0:
                    self.opponent_chips -= call_amount
                    self.opponent_bet += call_amount
                    self.pot += call_amount
                    self.action_history.append(("Adversaire", "CALL", call_amount))
                
                # showdown
                return self.proceed_to_showdown()
            
            else:
                # Autres actions (unlikely après ALL_IN mais pour sécurité)
                return self.get_state(), 0, self.done, {}
                
        # Cas général: continuer le jeu normalement
        return self.get_state(), 0, self.done, {}

    # procéder au showdown si les deux joueurs sont ALL_IN ou si le round est terminé
    def proceed_to_showdown(self) -> Tuple[np.ndarray, float, bool, Dict]:
        # Révéler toutes les cartes communautaires restantes
        while len(self.community_cards) < 5:
            self.community_cards.append(self.deck.draw(1)[0])
        
        # Aller au showdown
        winner, player_hand_strength, opponent_hand_strength = self.evaluate_hands()
        reward = self.calculate_final_reward(winner)
        self.done = True
        
        return self.get_state(), reward, True, {
            'winner': winner,
            'player_hand': player_hand_strength,
            'opponent_hand': opponent_hand_strength
        }

    def evaluate_hands(self) -> Tuple[str, float, float]:
        player_strength = self.get_hand_strength(self.player_cards)
        opponent_strength = self.get_hand_strength(self.opponent_cards)
        
        if player_strength > opponent_strength:
            return "player", player_strength, opponent_strength
        elif player_strength < opponent_strength:
            return "opponent", player_strength, opponent_strength
        else:
            return "tie", player_strength, opponent_strength

    def calculate_final_reward(self, winner: str) -> float:
        if winner == "player":
            return self.pot - self.player_bet
        elif winner == "opponent":
            return -self.player_bet
        else:  # égalité
            return 0

    def get_legal_actions(self) -> List[int]:
        if self.player_chips == 0:
            return []  # Aucune action possible si le joueur n'a plus de jetons
        
        legal_actions = []
        
        # CHECK: possible si pas de mise adversaire ou si on a déjà égalisé
        if self.opponent_bet <= self.player_bet:
            legal_actions.append(PokerAction.CHECK.value)
        
        # CALL: possible s'il y a une mise adversaire à égaliser
        call_amount = max(0, self.opponent_bet - self.player_bet)
        if call_amount > 0 and call_amount <= self.player_chips:
            legal_actions.append(PokerAction.CALL.value)
        
        # RAISE_SMALL: possible si on a assez de jetons
        raise_small_total = call_amount + config.POKER.SMALL_RAISE
        if raise_small_total <= self.player_chips:
            legal_actions.append(PokerAction.RAISE_SMALL.value)
        
        # RAISE_BIG: possible si on a assez de jetons  
        raise_big_total = call_amount + config.POKER.BIG_RAISE
        if raise_big_total <= self.player_chips:
            legal_actions.append(PokerAction.RAISE_BIG.value)
        
        # ALL_IN: toujours possible si on a des jetons
        if self.player_chips > 0:
            legal_actions.append(PokerAction.ALL_IN.value)
        
        # FOLD: toujours possible (sauf si CHECK est possible)
        if PokerAction.CHECK.value not in legal_actions:
            legal_actions.append(PokerAction.FOLD.value)
        
        return legal_actions
