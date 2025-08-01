# TODO: multi agent
# # pas encore prêt

# import numpy as np
# import random
# from enum import Enum
# from typing import List, Tuple, Dict, Optional
# from treys import Card as TreysCard, Evaluator, Deck
# from config import config
# from poker_game import PokerAction


# class MultiAgentPokerEnv:
#     """Environnement de poker Texas Hold'em pour deux agents DQN"""
    
#     def __init__(self):
#         self.evaluator = Evaluator()
#         self.num_actions = config.POKER.NUM_ACTIONS
#         self.state_size = config.POKER.STATE_SIZE
#         self.reset()
    
#     def reset(self) -> Tuple[np.ndarray, np.ndarray]:
#         """Initialise une nouvelle partie et retourne les états des deux agents"""
#         self.deck = Deck()
#         self.deck.shuffle()
        
#         # Distribution des cartes
#         self.agent1_cards = [self.deck.draw(1)[0], self.deck.draw(1)[0]]
#         self.agent2_cards = [self.deck.draw(1)[0], self.deck.draw(1)[0]]
#         self.community_cards = []
        
#         # État initial
#         self.pot = config.POKER.SMALL_BLIND + config.POKER.BIG_BLIND
#         self.agent1_chips = config.POKER.INITIAL_CHIPS
#         self.agent2_chips = config.POKER.INITIAL_CHIPS
#         self.betting_round = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        
#         # Alternance des positions (Small Blind / Big Blind)
#         if random.random() < 0.5:
#             self.agent1_bet = config.POKER.SMALL_BLIND   # Agent 1 = Small blind
#             self.agent2_bet = config.POKER.BIG_BLIND     # Agent 2 = Big blind
#             self.agent1_position = "SB"
#             self.agent2_position = "BB" 
#             self.current_player = 1  # Agent 1 commence (SB agit en premier preflop)
#         else:
#             self.agent1_bet = config.POKER.BIG_BLIND     # Agent 1 = Big blind
#             self.agent2_bet = config.POKER.SMALL_BLIND   # Agent 2 = Small blind
#             self.agent1_position = "BB"
#             self.agent2_position = "SB"
#             self.current_player = 2  # Agent 2 commence (SB agit en premier preflop)
        
#         # Déduire les blinds des stacks
#         self.agent1_chips -= self.agent1_bet
#         self.agent2_chips -= self.agent2_bet
        
#         # États du jeu
#         self.done = False
#         self.winner = None
#         self.last_action_agent1 = None
#         self.last_action_agent2 = None
        
#         return self._get_state_agent1(), self._get_state_agent2()
    
#     def get_betting_round_name(self) -> str:
#         """Retourne le nom du round de mise actuel"""
#         rounds = ["Pre-flop", "Flop", "Turn", "River", "Showdown"]
#         return rounds[min(self.betting_round, 4)]
    
#     def _get_hand_strength(self, hole_cards: List[int], community_cards: Optional[List[int]] = None) -> float:
#         """Calcule la force de la main avec Treys"""
#         if community_cards is None:
#             community_cards = self.community_cards
        
#         if len(community_cards) < 3:
#             # Pre-flop : évaluation simplifiée
#             return self._get_preflop_strength(hole_cards)
        
#         all_cards = hole_cards + community_cards
#         if len(all_cards) < 5:
#             return self._get_preflop_strength(hole_cards) * 1000
        
#         # Évaluation avec treys (plus le score est bas, meilleure est la main)
#         hand_rank = self.evaluator.evaluate(all_cards[:5], all_cards[5:7] if len(all_cards) == 7 else [])
        
#         # Conversion : treys donne un score où 1 = meilleur, 7462 = pire
#         normalized_strength = 7463 - hand_rank
#         return normalized_strength
    
#     def _get_preflop_strength(self, hole_cards: List[int]) -> float:
#         """Évaluation preflop simplifiée"""
#         ranks = [TreysCard.get_rank_int(card) for card in hole_cards]
#         suits = [TreysCard.get_suit_int(card) for card in hole_cards]
        
#         # Paire
#         if ranks[0] == ranks[1]:
#             return 0.7 + (ranks[0] / 12.0) * 0.3
        
#         # Suited
#         if suits[0] == suits[1]:
#             high_card = max(ranks) / 12.0
#             return 0.4 + high_card * 0.3
        
#         # Offsuit
#         high_card = max(ranks) / 12.0
#         return 0.1 + high_card * 0.4
    
#     def _get_hand_class(self, hole_cards: List[int], community_cards: Optional[List[int]] = None) -> str:
#         """Retourne la classe de la main"""
#         if community_cards is None:
#             community_cards = self.community_cards
            
#         if len(community_cards) < 3:
#             return "Cartes hautes"
        
#         all_cards = hole_cards + community_cards
#         if len(all_cards) < 5:
#             return "Cartes hautes"
        
#         # Utiliser Treys pour obtenir la classe de main
#         hand_rank = self.evaluator.evaluate(all_cards[:5], all_cards[5:7] if len(all_cards) == 7 else [])
#         rank_class = self.evaluator.get_rank_class(hand_rank)
        
#         classes = {
#             1: "Quinte flush royale",
#             2: "Quinte flush", 
#             3: "Carré",
#             4: "Full house",
#             5: "Couleur",
#             6: "Quinte",
#             7: "Brelan",
#             8: "Double paire",
#             9: "Paire"
#         }
        
#         return classes.get(rank_class, "Cartes hautes")
    
#     def _get_state_agent1(self) -> np.ndarray:
#         """Retourne l'état du jeu du point de vue de l'agent 1"""
#         return self._get_state_for_agent(1)
    
#     def _get_state_agent2(self) -> np.ndarray:
#         """Retourne l'état du jeu du point de vue de l'agent 2"""
#         return self._get_state_for_agent(2)
    
#     def _get_state_for_agent(self, agent_id: int) -> np.ndarray:
#         """Génère l'état pour un agent spécifique"""
#         if agent_id == 1:
#             my_cards = self.agent1_cards
#             my_chips = self.agent1_chips
#             my_bet = self.agent1_bet
#             my_position = self.agent1_position
#             opp_chips = self.agent2_chips
#             opp_bet = self.agent2_bet
#         else:
#             my_cards = self.agent2_cards
#             my_chips = self.agent2_chips
#             my_bet = self.agent2_bet
#             my_position = self.agent2_position
#             opp_chips = self.agent1_chips
#             opp_bet = self.agent1_bet
        
#         # Force de la main
#         hand_strength = self._get_hand_strength(my_cards)
#         normalized_strength = hand_strength / 7463.0 if hand_strength > 1 else hand_strength
        
#         # Informations sur les cartes privées
#         card1_rank = TreysCard.get_rank_int(my_cards[0]) / 12.0
#         card2_rank = TreysCard.get_rank_int(my_cards[1]) / 12.0
#         is_suited = float(TreysCard.get_suit_int(my_cards[0]) == TreysCard.get_suit_int(my_cards[1]))
#         is_pair = float(TreysCard.get_rank_int(my_cards[0]) == TreysCard.get_rank_int(my_cards[1]))
        
#         # Normalisation avec les paramètres de configuration
#         max_chips = config.POKER.INITIAL_CHIPS * 2
#         max_bet = config.POKER.INITIAL_CHIPS
        
#         state = [
#             normalized_strength,                                         # Force de la main normalisée
#             self.pot / max_chips,                                        # Taille du pot
#             my_chips / max_chips,                                        # Jetons de l'agent1
#             opp_chips / max_chips,                                       # Jetons de l'agent2
#             self.betting_round / 3.0,                                    # Round de mise
#             len(self.community_cards) / 5.0,                             # Nombre de cartes communes
#             my_bet / max_bet,                                            # Agent1 mise actuelle
#             opp_bet / max_bet,                                           # Agent2 mise actuelle
#             card1_rank,                                                  # Rang carte 1
#             card2_rank,                                                  # Rang carte 2
#             is_suited,                                                   # Suited?
#             is_pair,                                                     # Paire?
#             float(my_position == "BB"),                                  # Position (if big blind?)
#             abs(my_bet - opp_bet) / max_bet,                            # Différence de mise
#             min(my_chips, opp_chips) / max_chips                        # Stack le plus petit
#         ]
        
#         return np.array(state, dtype=np.float32)
    
#     def deal_community_cards(self) -> None:
#         """Distribue les cartes communes selon le round"""
#         if self.betting_round == 1 and len(self.community_cards) == 0:  # Flop
#             self.community_cards.extend([self.deck.draw(1)[0] for _ in range(3)])
#         elif self.betting_round == 2 and len(self.community_cards) == 3:  # Turn
#             self.community_cards.append(self.deck.draw(1)[0])
#         elif self.betting_round == 3 and len(self.community_cards) == 4:  # River
#             self.community_cards.append(self.deck.draw(1)[0])
    
#     def step(self, action_idx: int) -> Tuple[np.ndarray, np.ndarray, float, float, bool, Dict]:
#         """
#         Exécute une action pour l'agent actuel
#         Retourne: (state_agent1, state_agent2, reward_agent1, reward_agent2, done, info)
#         """
#         if self.done:
#             return self._get_state_agent1(), self._get_state_agent2(), 0, 0, True, {}
        
#         action = PokerAction(action_idx)
#         current_agent = self.current_player
        
#         # Exécuter l'action de l'agent actuel
#         reward1, reward2 = self._execute_action(current_agent, action)
        
#         # Changer de joueur pour le prochain tour
#         self._switch_player()
        
#         return self._get_state_agent1(), self._get_state_agent2(), reward1, reward2, self.done, {"winner": self.winner}
    
#     def _execute_action(self, agent_id: int, action: PokerAction) -> Tuple[float, float]:
#         """Exécute l'action d'un agent et retourne les récompenses (agent1, agent2)"""
        
#         # Initialiser les récompenses par défaut
#         reward1, reward2 = 0, 0
        
#         if agent_id == 1:
#             # Stocker l'action pour l'affichage
#             self.last_action_agent1 = action
            
#             if action == PokerAction.FOLD:
#                 self.done = True
#                 reward1 = -self.agent1_bet
#                 reward2 = self.pot - self.agent2_bet
#                 self.winner = "agent2"
#                 return reward1, reward2  # Retourner immédiatement
                
#             elif action == PokerAction.CALL:
#                 call_amount = max(0, self.agent2_bet - self.agent1_bet)
#                 if call_amount <= self.agent1_chips:
#                     self.agent1_chips -= call_amount
#                     self.agent1_bet += call_amount
#                     self.pot += call_amount
                    
#             elif action == PokerAction.RAISE_SMALL:
#                 raise_amount = max(config.POKER.SMALL_RAISE, self.agent2_bet - self.agent1_bet + config.POKER.SMALL_RAISE)
#                 if raise_amount <= self.agent1_chips:
#                     self.agent1_chips -= raise_amount
#                     self.agent1_bet += raise_amount
#                     self.pot += raise_amount
                    
#             elif action == PokerAction.RAISE_BIG:
#                 raise_amount = max(config.POKER.BIG_RAISE, self.agent2_bet - self.agent1_bet + config.POKER.BIG_RAISE)
#                 if raise_amount <= self.agent1_chips:
#                     self.agent1_chips -= raise_amount
#                     self.agent1_bet += raise_amount
#                     self.pot += raise_amount
                    
#             elif action == PokerAction.ALL_IN:
#                 all_in_amount = self.agent1_chips
#                 self.agent1_bet += all_in_amount
#                 self.pot += all_in_amount
#                 self.agent1_chips = 0
                
#         else:  # agent_id == 2
#             # Stocker l'action pour l'affichage
#             self.last_action_agent2 = action
            
#             if action == PokerAction.FOLD:
#                 self.done = True
#                 reward1 = self.pot - self.agent1_bet
#                 reward2 = -self.agent2_bet
#                 self.winner = "agent1"
#                 return reward1, reward2  # Retourner immédiatement
                
#             elif action == PokerAction.CALL:
#                 call_amount = max(0, self.agent1_bet - self.agent2_bet)
#                 if call_amount <= self.agent2_chips:
#                     self.agent2_chips -= call_amount
#                     self.agent2_bet += call_amount
#                     self.pot += call_amount
                    
#             elif action == PokerAction.RAISE_SMALL:
#                 raise_amount = max(config.POKER.SMALL_RAISE, self.agent1_bet - self.agent2_bet + config.POKER.SMALL_RAISE)
#                 if raise_amount <= self.agent2_chips:
#                     self.agent2_chips -= raise_amount
#                     self.agent2_bet += raise_amount
#                     self.pot += raise_amount
                    
#             elif action == PokerAction.RAISE_BIG:
#                 raise_amount = max(config.POKER.BIG_RAISE, self.agent1_bet - self.agent2_bet + config.POKER.BIG_RAISE)
#                 if raise_amount <= self.agent2_chips:
#                     self.agent2_chips -= raise_amount
#                     self.agent2_bet += raise_amount
#                     self.pot += raise_amount
                    
#             elif action == PokerAction.ALL_IN:
#                 all_in_amount = self.agent2_chips
#                 self.agent2_bet += all_in_amount
#                 self.pot += all_in_amount
#                 self.agent2_chips = 0
        
#         # Vérifier si on passe au round suivant ou showdown (seulement si la partie n'est pas terminée)
#         if not self.done and self.agent1_bet == self.agent2_bet:
#             self.betting_round += 1
#             self.deal_community_cards()
            
#             if self.betting_round > 3:  # Showdown
#                 self.done = True
#                 agent1_strength = self._get_hand_strength(self.agent1_cards)
#                 agent2_strength = self._get_hand_strength(self.agent2_cards)
                
#                 if agent1_strength > agent2_strength:
#                     reward1 = self.pot - self.agent1_bet
#                     reward2 = -self.agent2_bet
#                     self.winner = "agent1"
#                 elif agent1_strength < agent2_strength:
#                     reward1 = -self.agent1_bet
#                     reward2 = self.pot - self.agent2_bet
#                     self.winner = "agent2"
#                 else:  # Égalité
#                     reward1 = reward2 = 0
#                     self.winner = "tie"
        
#         return reward1, reward2
    
#     def _switch_player(self):
#         """Change le joueur actuel"""
#         self.current_player = 2 if self.current_player == 1 else 1
    
#     def get_agent_hand_info(self, agent_id: int) -> Tuple[float, str]:
#         """Retourne les informations sur la main d'un agent"""
#         if agent_id == 1:
#             cards = self.agent1_cards
#         else:
#             cards = self.agent2_cards
            
#         strength = self._get_hand_strength(cards)
#         hand_class = self._get_hand_class(cards)
#         return strength, hand_class
    
#     def is_agent_turn(self, agent_id: int) -> bool:
#         """Vérifie si c'est le tour de l'agent spécifié"""
#         return self.current_player == agent_id
