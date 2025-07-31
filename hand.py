import random
from enum import Enum
from typing import List, Optional, Tuple

# ----------- CARTES ---------------

SUITS = ['s', 'h', 'd', 'c']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


class Card:
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}{self.suit}"


class Deck:
    def __init__(self):
        self.cards = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        random.shuffle(self.cards)

    def deal(self, n: int) -> List[Card]:
        return [self.cards.pop() for _ in range(n)]


# ----------- ACTIONS ---------------

class ActionType(Enum):
    FOLD = 'fold'
    CALL = 'call'
    CHECK = 'check'
    BET = 'bet'
    RAISE = 'raise'
    ALL_IN = 'all-in'


class Action:
    def __init__(self, action_type: ActionType, amount: Optional[int] = None):
        self.action_type = action_type
        self.amount = amount

    def __repr__(self):
        if self.amount is not None:
            return f"{self.action_type.value}({self.amount})"
        return f"{self.action_type.value}"


# ----------- JOUEURS ---------------

class PlayerState:
    def __init__(self, name: str, stack: int):
        self.name = name
        self.hand: List[Card] = []
        self.stack = stack
        self.bet = 0
        self.folded = False
        self.all_in = False

    def __repr__(self):
        return (f"{self.name}: Hand={self.hand}, Stack={self.stack}, "
                f"Bet={self.bet}, Folded={self.folded}, All-in={self.all_in}")


# ----------- ETAT DU JEU ---------------

class GameState:
    def __init__(self, starting_stack: int = 1000):
        self.deck = Deck()
        self.players = [PlayerState("P1", starting_stack), PlayerState("P2", starting_stack)]
        self.community_cards: List[Card] = []
        self.pot = 0
        self.street = 'preflop'
        self.current_player = 0  # index of player to act
        self.history: List[Action] = []
        self.min_raise = 10
        self.last_bet = 0

    def deal_hole_cards(self):
        for player in self.players:
            player.hand = self.deck.deal(2)

    def deal_flop(self):
        self.deck.deal(1)  # burn
        self.community_cards += self.deck.deal(3)
        self.street = 'flop'

    def deal_turn(self):
        self.deck.deal(1)
        self.community_cards += self.deck.deal(1)
        self.street = 'turn'

    def deal_river(self):
        self.deck.deal(1)
        self.community_cards += self.deck.deal(1)
        self.street = 'river'

    def reset_bets(self):
        for p in self.players:
            p.bet = 0
        self.last_bet = 0

    def __repr__(self):
        return (f"Street: {self.street}, Pot: {self.pot}, "
                f"Board: {self.community_cards}, "
                f"P1: {self.players[0]}, P2: {self.players[1]}")


# ----------- NOEUD D'ARBRE POUR CFR/MCCFR ---------------

class Node:
    def __init__(self, state: GameState, parent=None):
        self.state = state
        self.parent = parent
        self.children: List[Tuple[Action, 'Node']] = []

    def add_child(self, action: Action, child_node: 'Node'):
        self.children.append((action, child_node))

    def is_terminal(self):
        active = [p for p in self.state.players if not p.folded and not p.all_in]
        return len(active) <= 1 or self.state.street == 'river' and self.betting_round_over()

    def betting_round_over(self):
        # à améliorer : ici on pourrait ajouter la logique "quand tous les joueurs ont callé ou sont all-in"
        return True

    def __repr__(self):
        return f"<Node | {self.state.street} | Pot: {self.state.pot} | Actions: {len(self.children)}>"
    
    def apply_action(self, action: Action):
        player = self.players[self.current_player]
        opponent = self.players[1 - self.current_player]
        
        if action.action_type == ActionType.FOLD:
            player.folded = True

        elif action.action_type == ActionType.CHECK:
            # Vérifie que le joueur peut checker (aucune mise à égaliser)
            if player.bet < opponent.bet:
                raise ValueError("Cannot check, must call or fold.")

        elif action.action_type == ActionType.CALL:
            to_call = opponent.bet - player.bet
            amount = min(to_call, player.stack)
            player.stack -= amount
            player.bet += amount
            self.pot += amount
            if player.stack == 0:
                player.all_in = True

        elif action.action_type in {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}:
            if action.amount is None or action.amount <= 0:
                raise ValueError("Invalid bet/raise amount.")

            amount = min(action.amount, player.stack)
            to_call = opponent.bet - player.bet
            total_bet = to_call + amount

            player.stack -= total_bet
            player.bet += total_bet
            self.pot += total_bet
            self.last_bet = amount

            if player.stack == 0:
                player.all_in = True

        else:
            raise ValueError(f"Unsupported action type: {action.action_type}")

        self.history.append(action)
        # Passe au joueur suivant (tour de table)
        self.current_player = 1 - self.current_player
