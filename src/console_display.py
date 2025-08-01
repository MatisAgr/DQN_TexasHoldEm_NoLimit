# gestion des affichage stylé en console

import os
from colorama import init, Fore, Back, Style
from treys import Card as TreysCard
from typing import List
from poker_game import PokerAction, TreysPokerEnv
from config import config

init(autoreset=True) # reset de la couleur après chaque print


class PokerConsole:
    """console pour afficher le jeu"""
    
    def clear_screen() -> None:
        """Efface l'écran de la console"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(title: str) -> None:
        """entete trop stylé"""
        print(f"\n{Fore.CYAN}{'=' * 80}")
        print(f"{Fore.YELLOW}{title.center(80)}")
        print(f"{Fore.CYAN}{'=' * 80}")
    
    def print_cards(cards: List[int], label: str = "Cartes") -> None:
        """Affiche les cartes avec des couleurs appropriées"""
        cards_str = []
        for card in cards:
            card_str = TreysCard.int_to_pretty_str(card)
            # Colorier selon la couleur
            if '♥' in card_str or '♦' in card_str:
                cards_str.append(f"{Fore.RED}{card_str}{Style.RESET_ALL}")
            else:
                cards_str.append(f"{Fore.WHITE}{card_str}{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}{label}: {' '.join(cards_str)}{Style.RESET_ALL}")
    
    def print_game_state(env: TreysPokerEnv) -> None:
        """Affiche l'état actuel de la partie avec positions et joueur actuel"""
        # Indicateur du joueur qui doit jouer
        current_turn = getattr(env, 'current_turn', 'player')
        if current_turn == 'player':
            PokerConsole.print_player_turn("🤖 IA")
        else:
            PokerConsole.print_player_turn("🎲 ADVERSAIRE")
        
        # Informations sur les positions et blinds
        player_pos = getattr(env, 'player_position', 'SB')
        opponent_pos = getattr(env, 'opponent_position', 'BB')
        
        print(f"\n{Fore.MAGENTA}┌─ ÉTAT DE LA PARTIE ───────────────────────────────────────────────────┐")
        print(f"│ Pot: {Fore.YELLOW}{env.pot:>6}{Style.RESET_ALL} jetons\t\t│\tRound: {Fore.CYAN}{env.get_betting_round_name():<12}{Style.RESET_ALL}\t│")
        
        # Affichage avec positions
        pos_color_player = Fore.YELLOW if player_pos == "BB" else Fore.CYAN
        pos_color_opponent = Fore.YELLOW if opponent_pos == "BB" else Fore.CYAN
        
        print(f"│ 🤖 IA ({pos_color_player}{player_pos}{Style.RESET_ALL}): {Fore.GREEN}{env.player_chips:>4}{Style.RESET_ALL} jetons\t│\t🎲 Adv ({pos_color_opponent}{opponent_pos}{Style.RESET_ALL}): {Fore.RED}{env.opponent_chips:>4}{Style.RESET_ALL} jetons\t│")
        print(f"│ IA mise: {Fore.BLUE}{env.player_bet:>6}{Style.RESET_ALL} jetons\t│\tMise Adv: {Fore.BLUE}{env.opponent_bet:>6}{Style.RESET_ALL} jetons\t│")
        print(f"{Fore.MAGENTA}└───────────────────────────────────────────────────────────────────────┘{Style.RESET_ALL}")
        
        # Afficher les dernières actions des joueurs
        PokerConsole.print_last_actions(env)
        
        # Afficher l'historique complet si demandé (optionnel)
        if hasattr(env, 'show_full_history') and env.show_full_history:
            PokerConsole.print_complete_action_history(env.action_history)
    
    def print_player_action_live(player: str, action: str, amount: int = 0) -> None:
        """Affiche l'action d'un joueur en temps réel avec un style distinct"""
        action_colors = {
            "FOLD": Fore.RED,
            "CHECK": Fore.BLUE,
            "CALL": Fore.YELLOW,
            "RAISE_SMALL": Fore.CYAN,
            "RAISE_BIG": Fore.MAGENTA,
            "ALL_IN": Fore.RED + Style.BRIGHT
        }
        color = action_colors.get(action, Fore.WHITE)
        player_icon = "🤖" if player == "IA" else "🎲"
        
        print(f"\n{Fore.WHITE}{'=' * 80}")
        print(f"{Fore.YELLOW}➤ ACTION EN COURS{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'=' * 80}")
        
        if amount > 0:
            print(f"\n{Fore.WHITE}   {player_icon} {player} joue: {color}{action} ({amount} jetons){Style.RESET_ALL}")
        else:
            print(f"\n{Fore.WHITE}   {player_icon} {player} joue: {color}{action}{Style.RESET_ALL}")
        
        print(f"{Fore.WHITE}{'=' * 80}{Style.RESET_ALL}\n")
    
    def print_action(player: str, action: PokerAction, amount: int = 0) -> None:
        """Affiche l'action d'un joueur avec des couleurs"""
        action_colors = {
            "FOLD": Fore.RED,
            "CHECK": Fore.BLUE,
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
    
    def print_player_turn(current_player: str) -> None:
        """Affiche clairement qui doit jouer"""
        print(f"\n{Fore.YELLOW}{'🎯 ' + current_player + ' DOIT JOUER':^80}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'─' * 80}{Style.RESET_ALL}")
    
    def print_player_move(player: str, action: str, amount: int = 0) -> None:
        """Affiche l'action d'un joueur de manière plus visible"""
        action_colors = {
            "FOLD": Fore.RED,
            "CHECK": Fore.BLUE,
            "CALL": Fore.YELLOW,
            "RAISE_SMALL": Fore.CYAN,
            "RAISE_BIG": Fore.MAGENTA,
            "ALL_IN": Fore.RED + Style.BRIGHT
        }
        color = action_colors.get(action, Fore.WHITE)
        
        if amount > 0:
            print(f"\n{Fore.WHITE}➤ {player}: {color}{action} ({amount} jetons){Style.RESET_ALL}")
        else:
            print(f"\n{Fore.WHITE}➤ {player}: {color}{action}{Style.RESET_ALL}")
    
    def print_game_state_separator() -> None:
        """Séparateur visuel entre les phases du jeu"""
        print(f"\n{Fore.CYAN}{'─' * 80}{Style.RESET_ALL}")
    
    def print_blinds_info(player_position: str, opponent_position: str, player_blind: int, opponent_blind: int) -> None:
        """Affiche les informations sur les blinds"""
        print(f"\n{Fore.MAGENTA}┌─ POSITIONS ET BLINDS ──────────────────────────────────────────────────┐")
        print(f"│ 🤖 IA: {Fore.CYAN}{player_position:<2}{Style.RESET_ALL} ({player_blind} jetons)  │  🎲 Adversaire: {Fore.CYAN}{opponent_position:<2}{Style.RESET_ALL} ({opponent_blind} jetons)  │")
        print(f"{Fore.MAGENTA}└───────────────────────────────────────────────────────────────────────┘{Style.RESET_ALL}")
    
    def print_last_actions(env: TreysPokerEnv) -> None:
        """Affiche les dernières actions des deux joueurs si disponibles"""
        if not hasattr(env, 'action_history') or not env.action_history:
            return
        
        # Prendre les 2 dernières actions (joueur et adversaire)
        recent_actions = env.action_history[-2:] if len(env.action_history) >= 2 else env.action_history
        
        if not recent_actions:
            return
            
        print(f"\n{Fore.GREEN}┌─ DERNIÈRES ACTIONS ────────────────────────────────────────────────────┐")
        
        for i, (player, action, amount) in enumerate(recent_actions):
            action_colors = {
                "FOLD": Fore.RED,
                "CHECK": Fore.BLUE,
                "CALL": Fore.YELLOW,
                "RAISE_SMALL": Fore.CYAN,
                "RAISE_BIG": Fore.MAGENTA,
                "ALL_IN": Fore.RED + Style.BRIGHT
            }
            color = action_colors.get(action, Fore.WHITE)
            player_icon = "🤖" if player == "IA" else "🎲"
            
            # Marquer la plus récente
            is_most_recent = (i == len(recent_actions) - 1)
            recent_indicator = f" {Fore.GREEN}← DERNIÈRE{Style.RESET_ALL}" if is_most_recent else ""
            
            if amount > 0:
                print(f"│ {player_icon} {player}: {color}{action} ({amount} jetons){Style.RESET_ALL}{recent_indicator}")
            else:
                print(f"│ {player_icon} {player}: {color}{action}{Style.RESET_ALL}{recent_indicator}")
        
        print(f"{Fore.GREEN}└───────────────────────────────────────────────────────────────────────┘{Style.RESET_ALL}")
    
    def print_complete_action_history(action_history: List[tuple]) -> None:
        """Affiche l'historique complet des actions de la partie"""
        if not action_history:
            return
            
        print(f"\n{Fore.CYAN}┌─ HISTORIQUE ACTIONS ───────────────────────────────────────────────────┐")
        print(f"│ {Fore.WHITE}Chronologie complète de la partie:{Style.RESET_ALL}")
        
        for i, (player, action, amount) in enumerate(action_history, 1):
            action_colors = {
                "FOLD": Fore.RED,
                "CHECK": Fore.BLUE,
                "CALL": Fore.YELLOW,
                "RAISE_SMALL": Fore.CYAN,
                "RAISE_BIG": Fore.MAGENTA,
                "ALL_IN": Fore.RED + Style.BRIGHT
            }
            color = action_colors.get(action, Fore.WHITE)
            player_icon = "🤖" if player == "IA" else "🎲"
            
            # Marquer la dernière action comme "move actuel"
            is_current_move = (i == len(action_history))
            current_indicator = f" {Fore.GREEN}(move actuel){Style.RESET_ALL}" if is_current_move else ""
            
            if amount > 0:
                print(f"│ {i:2d}. {player_icon} {player}: {color}{action} ({amount} jetons){Style.RESET_ALL}{current_indicator}")
            else:
                print(f"│ {i:2d}. {player_icon} {player}: {color}{action}{Style.RESET_ALL}{current_indicator}")
        
        print(f"{Fore.CYAN}└───────────────────────────────────────────────────────────────────────┘{Style.RESET_ALL}")
    
    def print_player_cards_box(player_name: str, cards: List[int], strength: float, rank_class: str, 
                               chips: int, bet: int, position: str) -> None:
        """Affiche les informations d'un joueur dans un encadré séparé"""
        player_icon = "🤖" if "IA" in player_name else "🎲"
        position_color = Fore.YELLOW if position == "BB" else Fore.CYAN
        
        print(f"\n{Fore.GREEN}┌─ {player_icon} {player_name.upper()} ──────────────────────────────────────────────────────┐")
        
        # Cartes
        cards_str = []
        for card in cards:
            card_str = TreysCard.int_to_pretty_str(card)
            if '♥' in card_str or '♦' in card_str:
                cards_str.append(f"{Fore.RED}{card_str}{Style.RESET_ALL}")
            else:
                cards_str.append(f"{Fore.WHITE}{card_str}{Style.RESET_ALL}")
        
        print(f"│ Cartes: {' '.join(cards_str)}")
        
        # Force de la main
        strength_color = Fore.GREEN if strength > 4000 else Fore.YELLOW if strength > 2000 else Fore.RED
        print(f"│ Force: {strength_color}{strength}{Style.RESET_ALL} ({rank_class})")
        
        # Statistiques
        print(f"│ Position: {position_color}{position}{Style.RESET_ALL}  │  Jetons: {Fore.BLUE}{chips}{Style.RESET_ALL}  │  Mise: {Fore.YELLOW}{bet}{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}└─────────────────────────────────────────────────────────────────────────┘{Style.RESET_ALL}")
    
    def print_hand_strength(strength: float, rank_class: str) -> None:
        """Affiche la force de la main avec des couleurs"""
        strength_color = Fore.GREEN if strength > 4000 else Fore.YELLOW if strength > 2000 else Fore.RED
        print(f"Force de la main: {strength_color}{strength}{Style.RESET_ALL} ({rank_class})")
    
    def print_training_progress(episode: int, total_reward: float, epsilon: float, 
                              win_rate: float, avg_reward: float, avg_loss: float) -> None:
        """Affiche le progrès de l'entraînement"""
        win_color = Fore.GREEN if win_rate > 60 else Fore.YELLOW if win_rate > 40 else Fore.RED
        reward_color = Fore.GREEN if avg_reward > 0 else Fore.RED
        
        print(f"{Fore.CYAN}Episode {episode:4d}{Style.RESET_ALL} | "
              f"Recompense: {reward_color}{total_reward:6.1f}{Style.RESET_ALL} | "
              f"Epsilon: {Fore.BLUE}{epsilon:.3f}{Style.RESET_ALL} | "
              f"Victoires: {win_color}{win_rate:5.1f}%{Style.RESET_ALL} | "
              f"R.moy: {reward_color}{avg_reward:6.1f}{Style.RESET_ALL} | "
              f"Loss: {Fore.MAGENTA}{avg_loss:.4f}{Style.RESET_ALL}")
    
    def print_final_results(episodes: int, final_win_rate: float, 
                          final_avg_reward: float, final_avg_loss: float, 
                          memory_size: int) -> None:
        """Affiche les statistiques finales de l'entraînement"""
        print("\n" + "=" * 70)
        print("ENTRAINEMENT TERMINE!")
        print("=" * 70)
        
        print(f"{Fore.GREEN}Statistiques finales:")
        print(f"   Episodes totaux: {Fore.CYAN}{episodes}{Style.RESET_ALL}")
        print(f"   Taux de victoire final (200 derniers): {Fore.YELLOW}{final_win_rate:.1f}%{Style.RESET_ALL}")
        print(f"   Recompense moyenne finale: {Fore.BLUE}{final_avg_reward:.2f}{Style.RESET_ALL}")
        print(f"   Loss moyenne finale: {Fore.MAGENTA}{final_avg_loss:.4f}{Style.RESET_ALL}")
        print(f"   Memoire de replay: {Fore.RED}{memory_size}{Style.RESET_ALL} transitions")
    
    def print_test_results(test_wins: int, total_tests: int, test_rewards: List[float]) -> None:
        """Affiche les résultats des tests"""
        test_win_rate = test_wins / total_tests * 100
        
        PokerConsole.print_header("RESULTATS DU TEST FINAL")
        print(f"{Fore.GREEN}Victoires: {Fore.YELLOW}{test_wins}/{total_tests}{Style.RESET_ALL} ({Fore.CYAN}{test_win_rate:.1f}%{Style.RESET_ALL})")
        print(f"{Fore.BLUE}Recompense moyenne: {Fore.YELLOW}{sum(test_rewards)/len(test_rewards):.2f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Meilleure recompense: {Fore.YELLOW}{max(test_rewards):.1f}{Style.RESET_ALL}")
        print(f"{Fore.RED}Pire recompense: {Fore.YELLOW}{min(test_rewards):.1f}{Style.RESET_ALL}")
        
        # Message final basé sur les performances
        if test_win_rate > 60:
            print(f"\n{Fore.GREEN}Nickel l'agent performe bien{Style.RESET_ALL}")
        elif test_win_rate > 40:
            print(f"\n{Fore.YELLOW}Ok tier sur la performance{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}Bof la performance là{Style.RESET_ALL}")
    
    def render_game(env: TreysPokerEnv, show_opponent_cards: bool = False) -> None:
        """Affiche l'état complet du jeu avec une interface améliorée"""
        PokerConsole.print_header(f"POKER DQN - {env.get_betting_round_name()}")
        
        # Informations sur les cartes privées dans des encadrés séparés
        player_strength, player_class = env.get_player_hand_info()
        player_pos = getattr(env, 'player_position', 'SB')
        PokerConsole.print_player_cards_box("IA", env.player_cards, player_strength, player_class, 
                                           env.player_chips, env.player_bet, player_pos)
        
        # Cartes de l'adversaire (optionnel)
        if show_opponent_cards:
            opp_strength, opp_class = env.get_opponent_hand_info()
            opponent_pos = getattr(env, 'opponent_position', 'BB')
            PokerConsole.print_player_cards_box("Adversaire", env.opponent_cards, opp_strength, opp_class,
                                               env.opponent_chips, env.opponent_bet, opponent_pos)
        
        # Cartes communes
        if env.community_cards:
            print(f"\n{Fore.YELLOW}┌─ 🃏 BOARD (CARTES COMMUNES) ────────────────────────────────────────────┐")
            cards_str = []
            for card in env.community_cards:
                card_str = TreysCard.int_to_pretty_str(card)
                if '♥' in card_str or '♦' in card_str:
                    cards_str.append(f"{Fore.RED}{card_str}{Style.RESET_ALL}")
                else:
                    cards_str.append(f"{Fore.WHITE}{card_str}{Style.RESET_ALL}")
            print(f"│ Cartes: {' '.join(cards_str)}")
            print(f"{Fore.YELLOW}└─────────────────────────────────────────────────────────────────────────┘{Style.RESET_ALL}")
        
        # État du jeu avec historique
        PokerConsole.print_game_state(env)
        
        # Séparateur avant les résultats finaux
        if env.done:
            PokerConsole.print_game_state_separator()
            
            # Afficher l'historique complet à la fin de la partie
            if hasattr(env, 'action_history') and env.action_history:
                PokerConsole.print_complete_action_history(env.action_history)
            
            if env.winner == "player":
                print(f"\n{Fore.GREEN}🎉 VICTOIRE ! Agent IA gagne {env.pot - env.player_bet} jetons {Style.RESET_ALL}")
            elif env.winner == "opponent":
                print(f"\n{Fore.RED}💀 DÉFAITE ! Agent IA perd {env.player_bet} jetons {Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}🤝 ÉGALITÉ ! Personne ne gagne {Style.RESET_ALL}")
                            
    def render_multi_agent_game(env, show_both_hands: bool = False) -> None:
        """Affiche l'état complet du jeu multi-agent"""
        PokerConsole.print_header(f"MULTI-AGENT POKER - {env.get_betting_round_name()}")
        
        # Cartes de l'agent 1
        PokerConsole.print_cards(env.agent1_cards, f"🤖 {config.TRAINING.AGENT_NAMES[0]} main")
        agent1_strength, agent1_class = env.get_agent_hand_info(1)
        PokerConsole.print_hand_strength(agent1_strength, agent1_class)
        
        # Cartes de l'agent 2 (optionnel)
        if show_both_hands:
            PokerConsole.print_cards(env.agent2_cards, f"🤖 {config.TRAINING.AGENT_NAMES[1]} main")
            agent2_strength, agent2_class = env.get_agent_hand_info(2)
            print(f"{Fore.BLUE}Force {config.TRAINING.AGENT_NAMES[1]}: {agent2_strength} ({agent2_class}){Style.RESET_ALL}")
        
        # Cartes communes
        if env.community_cards:
            PokerConsole.print_cards(env.community_cards, "🃏 Board")
        
        # État du jeu multi-agent
        PokerConsole.print_multi_agent_game_state(env)
        
        # Actions des agents
        if env.last_action_agent1:
            PokerConsole.print_action(f"🤖 {config.TRAINING.AGENT_NAMES[0]} move", env.last_action_agent1)
        
        if env.last_action_agent2:
            PokerConsole.print_action(f"🤖 {config.TRAINING.AGENT_NAMES[1]} move", env.last_action_agent2)
        
        # Tour actuel
        current_agent = config.TRAINING.AGENT_NAMES[env.current_player - 1]
        print(f"{Fore.YELLOW}🎯 Tour de: {current_agent}{Style.RESET_ALL}")
        
        # Résultat final
        if env.done:
            if env.winner == "agent1":
                print(f"\n{Fore.GREEN} Victoire {config.TRAINING.AGENT_NAMES[0]} gagne {env.pot - env.agent1_bet} jetons {Style.RESET_ALL}")
            elif env.winner == "agent2":
                print(f"\n{Fore.GREEN} Victoire {config.TRAINING.AGENT_NAMES[1]} gagne {env.pot - env.agent2_bet} jetons {Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}Personne ne gagne{Style.RESET_ALL}")
                        
    
    def print_multi_agent_game_state(env) -> None:
        """Affiche l'état du jeu multi-agent"""
        print(f"\n{Fore.MAGENTA}┌─ ÉTAT MULTI-AGENT ────────────────────────────────────────────────────┐")
        print(f"│ Pot: {Fore.YELLOW}{env.pot:>6}{Style.RESET_ALL} jetons\t\t│\tRound: {Fore.CYAN}{env.get_betting_round_name():<12}{Style.RESET_ALL}\t\t│")
        print(f"│ {config.TRAINING.AGENT_NAMES[0]} jetons: {Fore.GREEN}{env.agent1_chips:>6}{Style.RESET_ALL}\t│\t{config.TRAINING.AGENT_NAMES[1]} jetons: {Fore.RED}{env.agent2_chips:>6}{Style.RESET_ALL}\t│")
        print(f"│ {config.TRAINING.AGENT_NAMES[0]} mise: {Fore.BLUE}{env.agent1_bet:>6}{Style.RESET_ALL}\t│\t{config.TRAINING.AGENT_NAMES[1]} mise: {Fore.BLUE}{env.agent2_bet:>6}{Style.RESET_ALL}\t\t│")
        print(f"│ Position: {Fore.CYAN}{env.agent1_position}{Style.RESET_ALL}\t\t│\tPosition: {Fore.CYAN}{env.agent2_position}{Style.RESET_ALL}\t\t\t│")
        print(f"{Fore.MAGENTA}└───────────────────────────────────────────────────────────────────────┘{Style.RESET_ALL}")
    
    def print_multi_agent_training_progress(episode: int, agent1_reward: float, agent2_reward: float,
                                          agent1_epsilon: float, agent2_epsilon: float,
                                          agent1_win_rate: float, agent2_win_rate: float,
                                          avg_loss1: float, avg_loss2: float) -> None:
        """Affiche le progrès de l'entraînement multi-agent"""
        agent1_color = Fore.GREEN if agent1_win_rate > 50 else Fore.RED
        agent2_color = Fore.GREEN if agent2_win_rate > 50 else Fore.RED
        
        print(f"{Fore.CYAN}Episode {episode:4d}{Style.RESET_ALL}")
        print(f"  {config.TRAINING.AGENT_NAMES[0]}: Reward: {agent1_color}{agent1_reward:6.1f}{Style.RESET_ALL} | "
              f"Epsilon: {Fore.BLUE}{agent1_epsilon:.3f}{Style.RESET_ALL} | "
              f"Wins: {agent1_color}{agent1_win_rate:5.1f}%{Style.RESET_ALL} | "
              f"Loss: {Fore.MAGENTA}{avg_loss1:.4f}{Style.RESET_ALL}")
        print(f"  {config.TRAINING.AGENT_NAMES[1]}: Reward: {agent2_color}{agent2_reward:6.1f}{Style.RESET_ALL} | "
              f"Epsilon: {Fore.BLUE}{agent2_epsilon:.3f}{Style.RESET_ALL} | "
              f"Wins: {agent2_color}{agent2_win_rate:5.1f}%{Style.RESET_ALL} | "
              f"Loss: {Fore.MAGENTA}{avg_loss2:.4f}{Style.RESET_ALL}")
