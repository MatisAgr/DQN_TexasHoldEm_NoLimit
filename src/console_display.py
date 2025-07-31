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
        """Affiche l'état actuel de la partie"""
        print(f"\n{Fore.MAGENTA}┌─ ÉTAT DE LA PARTIE ───────────────────────────────────────────────────┐")
        print(f"│ Pot: {Fore.YELLOW}{env.pot:>6}{Style.RESET_ALL} jetons\t\t│\tRound: {Fore.CYAN}{env.get_betting_round_name():<12}{Style.RESET_ALL}\t\t│")
        print(f"│ IA jetons: {Fore.GREEN}{env.player_chips:>6}{Style.RESET_ALL} jetons\t│\tJetons Adv: {Fore.RED}{env.opponent_chips:>6}{Style.RESET_ALL} jetons\t│")
        print(f"│ IA mise: {Fore.BLUE}{env.player_bet:>6}{Style.RESET_ALL} jetons\t│\tMise Adv: {Fore.BLUE}{env.opponent_bet:>6}{Style.RESET_ALL} jetons\t\t│")
        print(f"{Fore.MAGENTA}└───────────────────────────────────────────────────────────────────────┘{Style.RESET_ALL}")
    
    def print_action(player: str, action: PokerAction, amount: int = 0) -> None:
        """Affiche l'action d'un joueur avec des couleurs"""
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
        """Affiche l'état complet du jeu"""
        PokerConsole.print_header(f"POKER DQN - {env.get_betting_round_name()}")
        
        # Cartes du joueur
        PokerConsole.print_cards(env.player_cards, "🤖 Agent IA main")
        
        # Force de la main du joueur
        player_strength, player_class = env.get_player_hand_info()
        PokerConsole.print_hand_strength(player_strength, player_class)
        
        # Cartes de l'adversaire (optionnel)
        if show_opponent_cards:
            PokerConsole.print_cards(env.opponent_cards, "🎲 Cartes adversaire")
            opp_strength, opp_class = env.get_opponent_hand_info()
            print(f"{Fore.RED}Force adversaire: {opp_strength} ({opp_class}){Style.RESET_ALL}")
        
        # Cartes communes
        if env.community_cards:
            PokerConsole.print_cards(env.community_cards, "🃏 Board")
        
        # État du jeu
        PokerConsole.print_game_state(env)
        
        # Dernière action du joueur
        if env.last_action:
            PokerConsole.print_action("🤖 Agent IA move", env.last_action)
        
        # Dernière action de l'adversaire 
        if hasattr(env, 'opponent_last_action') and env.opponent_last_action:
            PokerConsole.print_action("🎲 Adversaire move", env.opponent_last_action)
        
        # Résultat final
        if env.done:
            if env.winner == "player":
                print(f"\n{Fore.GREEN}Agent IA gagne {env.pot - env.player_bet} jetons {Style.RESET_ALL}")
            elif env.winner == "opponent":
                print(f"\n{Fore.RED}Agent IA perd {env.player_bet} jetons {Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}Personne ne gagne {Style.RESET_ALL}")
                        
        print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}")
    
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
                        
        print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}")
    
    @staticmethod  
    def print_multi_agent_game_state(env) -> None:
        """Affiche l'état du jeu multi-agent"""
        print(f"\n{Fore.MAGENTA}┌─ ÉTAT MULTI-AGENT ────────────────────────────────────────────────────┐")
        print(f"│ Pot: {Fore.YELLOW}{env.pot:>6}{Style.RESET_ALL} jetons\t\t│\tRound: {Fore.CYAN}{env.get_betting_round_name():<12}{Style.RESET_ALL}\t\t│")
        print(f"│ {config.TRAINING.AGENT_NAMES[0]} jetons: {Fore.GREEN}{env.agent1_chips:>6}{Style.RESET_ALL}\t│\t{config.TRAINING.AGENT_NAMES[1]} jetons: {Fore.RED}{env.agent2_chips:>6}{Style.RESET_ALL}\t│")
        print(f"│ {config.TRAINING.AGENT_NAMES[0]} mise: {Fore.BLUE}{env.agent1_bet:>6}{Style.RESET_ALL}\t│\t{config.TRAINING.AGENT_NAMES[1]} mise: {Fore.BLUE}{env.agent2_bet:>6}{Style.RESET_ALL}\t\t│")
        print(f"│ Position: {Fore.CYAN}{env.agent1_position}{Style.RESET_ALL}\t\t│\tPosition: {Fore.CYAN}{env.agent2_position}{Style.RESET_ALL}\t\t\t│")
        print(f"{Fore.MAGENTA}└───────────────────────────────────────────────────────────────────────┘{Style.RESET_ALL}")
    
    @staticmethod
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
