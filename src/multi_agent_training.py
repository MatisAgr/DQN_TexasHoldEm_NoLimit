# TODO: train multi agent
# # pas encore prêt


# import os
# import numpy as np
# import time
# from colorama import Fore, Style
# from config import config
# from multi_agent_poker import MultiAgentPokerEnv
# from console_display import PokerConsole
# from dqn_agent import DQNAgent
# from poker_game import PokerAction


# def train_multi_agent(env: MultiAgentPokerEnv, agent1: DQNAgent, agent2: DQNAgent, 
#                      episodes: int = None, show_game_every: int = None, show_both_hands: bool = None,
#                      train_frequency: int = None, target_update_frequency: int = None) -> None:
#     """
#     Entraîne deux agents DQN l'un contre l'autre
#     """
#     # Utilisation des paramètres de configuration par défaut si non spécifiés
#     if episodes is None:
#         episodes = config.TRAINING.EPISODES
#     if show_game_every is None:
#         show_game_every = config.TRAINING.SHOW_GAME_EVERY
#     if show_both_hands is None:
#         show_both_hands = config.TRAINING.SHOW_OPPONENT_CARDS
#     if train_frequency is None:
#         train_frequency = config.TRAINING.TRAIN_FREQUENCY
#     if target_update_frequency is None:
#         target_update_frequency = config.TRAINING.TARGET_UPDATE_FREQUENCY
    
#     # Historiques des agents
#     agent1_reward_history = []
#     agent1_win_history = []
#     agent1_loss_history = []
    
#     agent2_reward_history = []
#     agent2_win_history = []
#     agent2_loss_history = []
    
#     step_count = 0
    
#     # Affichage initial
#     PokerConsole.clear_screen()
#     PokerConsole.print_header("ENTRAINEMENT MULTI-AGENT DQN POKER")
#     print(f"{Fore.GREEN}Mode: IA vs IA - {config.TRAINING.AGENT_NAMES[0]} vs {config.TRAINING.AGENT_NAMES[1]}")
#     print(f"Episodes prévus: {episodes}")
#     print(f"Architecture: 2x Réseaux de neurones DQN indépendants")
#     print(f"Environnement: Poker Texas Hold'em avec évaluation Treys{Style.RESET_ALL}")
#     print(f"\nDébut de l'entraînement multi-agent...\n")
    
#     for episode in range(episodes):
#         # Reset de l'environnement
#         state1, state2 = env.reset()
        
#         episode_reward1 = 0
#         episode_reward2 = 0
#         episode_loss1 = 0
#         episode_loss2 = 0
#         steps_in_episode = 0
        
#         # Affichage de la partie si c'est le bon intervalle
#         show_this_game = (episode % show_game_every == 0 and episode > 0)
        
#         if show_this_game:
#             PokerConsole.clear_screen()
#             print(f"{Fore.YELLOW}Démonstration Multi-Agent - Episode {episode}{Style.RESET_ALL}")
#             PokerConsole.render_multi_agent_game(env, show_both_hands)
#             time.sleep(1)
        
#         while not env.done:
#             current_agent_id = env.current_player
            
#             # Sélection de l'action par l'agent actuel
#             if current_agent_id == 1:
#                 action = agent1.act(state1, training=True)
#                 current_state = state1
#                 current_agent = agent1
#                 agent_name = config.TRAINING.AGENT_NAMES[0]
#             else:
#                 action = agent2.act(state2, training=True)
#                 current_state = state2
#                 current_agent = agent2
#                 agent_name = config.TRAINING.AGENT_NAMES[1]
            
#             # Exécution de l'action
#             next_state1, next_state2, reward1, reward2, done, info = env.step(action)
            
#             # Affichage si démonstration
#             if show_this_game:
#                 action_enum = PokerAction(action)
#                 PokerConsole.print_action(f"🎯 {agent_name}", action_enum)
#                 time.sleep(0.8)
                
#                 PokerConsole.render_multi_agent_game(env, show_both_hands)
#                 if not done:
#                     time.sleep(1)
            
#             # Stockage des transitions pour les deux agents
#             agent1.store_transition(state1, action if current_agent_id == 1 else None, 
#                                   reward1, next_state1, done)
#             agent2.store_transition(state2, action if current_agent_id == 2 else None, 
#                                   reward2, next_state2, done)
            
#             # Mise à jour des récompenses cumulées
#             episode_reward1 += reward1
#             episode_reward2 += reward2
            
#             # Mise à jour des états
#             state1, state2 = next_state1, next_state2
#             steps_in_episode += 1
#             step_count += 1
            
#             # Entraînement périodique
#             if step_count % train_frequency == 0:
#                 if config.TRAINING.ALTERNATE_TRAINING:
#                     # Entraîner alternativement les agents
#                     if step_count // train_frequency % 2 == 0:
#                         loss1 = agent1.train_step()
#                         if loss1 is not None:
#                             episode_loss1 += loss1
#                     else:
#                         loss2 = agent2.train_step()
#                         if loss2 is not None:
#                             episode_loss2 += loss2
#                 else:
#                     # Entraîner les deux agents
#                     loss1 = agent1.train_step()
#                     loss2 = agent2.train_step()
#                     if loss1 is not None:
#                         episode_loss1 += loss1
#                     if loss2 is not None:
#                         episode_loss2 += loss2
        
#         # Fin de l'épisode - affichage résultat
#         if show_this_game:
#             print(f"\n{Fore.CYAN}Fin de la partie")
#             print(f"Récompense {config.TRAINING.AGENT_NAMES[0]}: {episode_reward1:.1f}")
#             print(f"Récompense {config.TRAINING.AGENT_NAMES[1]}: {episode_reward2:.1f}{Style.RESET_ALL}")
#             time.sleep(2)
        
#         # Mise à jour d'epsilon pour les deux agents
#         agent1.update_epsilon()
#         agent2.update_epsilon()
        
#         # Mise à jour des target networks
#         if episode % target_update_frequency == 0:
#             agent1.update_target_model()
#             agent2.update_target_model()
        
#         # Statistiques
#         agent1_reward_history.append(episode_reward1)
#         agent1_win_history.append(1 if episode_reward1 > episode_reward2 else 0)
#         agent1_loss_history.append(episode_loss1 / max(1, steps_in_episode))
        
#         agent2_reward_history.append(episode_reward2)
#         agent2_win_history.append(1 if episode_reward2 > episode_reward1 else 0)
#         agent2_loss_history.append(episode_loss2 / max(1, steps_in_episode))
        
#         # Affichage des résultats périodique
#         if episode % config.TRAINING.PROGRESS_EVERY == 0:
#             recent_wins1 = sum(agent1_win_history[-100:]) if len(agent1_win_history) >= 100 else sum(agent1_win_history)
#             recent_wins2 = sum(agent2_win_history[-100:]) if len(agent2_win_history) >= 100 else sum(agent2_win_history)
#             recent_episodes = min(100, len(agent1_win_history))
            
#             win_rate1 = recent_wins1 / recent_episodes * 100 if recent_episodes > 0 else 0
#             win_rate2 = recent_wins2 / recent_episodes * 100 if recent_episodes > 0 else 0
            
#             avg_loss1 = np.mean(agent1_loss_history[-100:]) if len(agent1_loss_history) >= 100 else (np.mean(agent1_loss_history) if agent1_loss_history else 0)
#             avg_loss2 = np.mean(agent2_loss_history[-100:]) if len(agent2_loss_history) >= 100 else (np.mean(agent2_loss_history) if agent2_loss_history else 0)
            
#             PokerConsole.print_multi_agent_training_progress(
#                 episode, episode_reward1, episode_reward2,
#                 agent1.epsilon, agent2.epsilon,
#                 win_rate1, win_rate2, avg_loss1, avg_loss2
#             )
        
#         # Sauvegarde périodique
#         if episode % config.TRAINING.SAVE_EVERY == 0 and episode > 0:
#             agent1.save_model(config.PATHS.AGENT1_EPISODE_TEMPLATE.format(episode))
#             agent2.save_model(config.PATHS.AGENT2_EPISODE_TEMPLATE.format(episode))
#             print(f"{Fore.GREEN}Modèles sauvegardés à l'épisode {episode}{Style.RESET_ALL}")
    
#     # Résultats finaux
#     final_win_rate1 = sum(agent1_win_history[-200:]) / min(200, len(agent1_win_history)) * 100
#     final_win_rate2 = sum(agent2_win_history[-200:]) / min(200, len(agent2_win_history)) * 100
#     final_avg_reward1 = np.mean(agent1_reward_history[-200:])
#     final_avg_reward2 = np.mean(agent2_reward_history[-200:])
    
#     PokerConsole.print_header("RÉSULTATS FINAUX MULTI-AGENT")
#     print(f"{Fore.GREEN}Statistiques finales:")
#     print(f"   Episodes totaux: {Fore.CYAN}{episodes}{Style.RESET_ALL}")
#     print(f"   {config.TRAINING.AGENT_NAMES[0]} - Victoires: {Fore.YELLOW}{final_win_rate1:.1f}%{Style.RESET_ALL} | Récompense moy: {Fore.BLUE}{final_avg_reward1:.2f}{Style.RESET_ALL}")
#     print(f"   {config.TRAINING.AGENT_NAMES[1]} - Victoires: {Fore.YELLOW}{final_win_rate2:.1f}%{Style.RESET_ALL} | Récompense moy: {Fore.BLUE}{final_avg_reward2:.2f}{Style.RESET_ALL}")
    
#     # Sauvegarde finale
#     agent1.save_model(config.PATHS.AGENT1_FINAL_MODEL)
#     agent2.save_model(config.PATHS.AGENT2_FINAL_MODEL)
#     print(f"{Fore.GREEN}Modèles finaux sauvegardés!{Style.RESET_ALL}")


# def test_multi_agent(env: MultiAgentPokerEnv, agent1: DQNAgent, agent2: DQNAgent, num_tests: int = 20) -> None:
#     """
#     Teste les deux agents entraînés l'un contre l'autre
#     """
#     PokerConsole.print_header("TEST MULTI-AGENT")
#     print(f"\n{Fore.YELLOW}Test des modèles sur {num_tests} parties (mode exploitation pur)...{Style.RESET_ALL}")
    
#     agent1_wins = 0
#     agent2_wins = 0
#     ties = 0
    
#     for test_episode in range(num_tests):
#         state1, state2 = env.reset()
#         episode_reward1 = 0
#         episode_reward2 = 0
        
#         # Affichage pour les 3 premières parties de test
#         show_test = test_episode < 3
        
#         if show_test:
#             print(f"\n{Fore.CYAN}Partie de test #{test_episode + 1}{Style.RESET_ALL}")
#             PokerConsole.render_multi_agent_game(env, show_both_hands=True)
#             time.sleep(1)
        
#         while not env.done:
#             current_agent_id = env.current_player
            
#             # Sélection d'action (mode test : pas d'exploration)
#             if current_agent_id == 1:
#                 action = agent1.act(state1, training=False)
#                 agent_name = config.TRAINING.AGENT_NAMES[0]
#             else:
#                 action = agent2.act(state2, training=False)
#                 agent_name = config.TRAINING.AGENT_NAMES[1]
            
#             if show_test:
#                 action_enum = PokerAction(action)
#                 PokerConsole.print_action(f"🎯 {agent_name}", action_enum)
#                 time.sleep(1)
            
#             state1, state2, reward1, reward2, done, info = env.step(action)
#             episode_reward1 += reward1
#             episode_reward2 += reward2
            
#             if show_test:
#                 PokerConsole.render_multi_agent_game(env, show_both_hands=True)
#                 if not done:
#                     time.sleep(1)
        
#         # Résultats
#         if episode_reward1 > episode_reward2:
#             agent1_wins += 1
#             result = f"{Fore.GREEN}{config.TRAINING.AGENT_NAMES[0]} GAGNE{Style.RESET_ALL}"
#         elif episode_reward2 > episode_reward1:
#             agent2_wins += 1
#             result = f"{Fore.GREEN}{config.TRAINING.AGENT_NAMES[1]} GAGNE{Style.RESET_ALL}"
#         else:
#             ties += 1
#             result = f"{Fore.YELLOW}ÉGALITÉ{Style.RESET_ALL}"
        
#         if show_test:
#             print(f"\n{Fore.CYAN}Fin de la partie de test - {result}")
#             print(f"Récompenses: {config.TRAINING.AGENT_NAMES[0]}={episode_reward1:.1f}, {config.TRAINING.AGENT_NAMES[1]}={episode_reward2:.1f}{Style.RESET_ALL}")
#             time.sleep(2)
        
#         print(f"Test {test_episode + 1:2d}: {result}")
    
#     # Résultats finaux
#     print(f"\n{Fore.CYAN}RÉSULTATS FINAUX:")
#     print(f"{config.TRAINING.AGENT_NAMES[0]}: {Fore.GREEN}{agent1_wins}{Style.RESET_ALL} victoires ({agent1_wins/num_tests*100:.1f}%)")
#     print(f"{config.TRAINING.AGENT_NAMES[1]}: {Fore.GREEN}{agent2_wins}{Style.RESET_ALL} victoires ({agent2_wins/num_tests*100:.1f}%)")
#     print(f"Égalités: {Fore.YELLOW}{ties}{Style.RESET_ALL} ({ties/num_tests*100:.1f}%){Style.RESET_ALL}")


# def main_multi_agent():
#     """Fonction principale pour l'entraînement multi-agent"""
#     print(f"{Fore.MAGENTA}🤖⚔️🤖 MODE MULTI-AGENT ACTIVÉ 🤖⚔️🤖{Style.RESET_ALL}")
    
#     # Utilisation des paramètres de configuration
#     np.random.seed(config.RANDOM_SEED)
    
#     # Création des répertoires nécessaires
#     os.makedirs(config.PATHS.CHECKPOINTS_DIR, exist_ok=True)
#     os.makedirs(config.PATHS.LOGS_DIR, exist_ok=True)
    
#     # Initialisation de l'environnement multi-agent
#     env = MultiAgentPokerEnv()
    
#     print(f"{Fore.CYAN}Environnement Multi-Agent Poker initialisé")
#     print(f"Dimension de l'état: {env.state_size}")
#     print(f"Nombre d'actions: {env.num_actions}")
#     print(f"Agents: {' vs '.join(config.TRAINING.AGENT_NAMES)}{Style.RESET_ALL}")
    
#     # Création des deux agents DQN
#     agent1 = DQNAgent(
#         state_size=env.state_size,
#         num_actions=env.num_actions,
#         learning_rate=config.DQN.LEARNING_RATE,
#         epsilon=config.DQN.EPSILON_START,
#         epsilon_min=config.DQN.EPSILON_MIN,
#         epsilon_decay=config.DQN.EPSILON_DECAY,
#         gamma=config.DQN.GAMMA,
#         memory_size=config.DQN.MEMORY_SIZE
#     )
    
#     agent2 = DQNAgent(
#         state_size=env.state_size,
#         num_actions=env.num_actions,
#         learning_rate=config.DQN.LEARNING_RATE,
#         epsilon=config.DQN.EPSILON_START,
#         epsilon_min=config.DQN.EPSILON_MIN,
#         epsilon_decay=config.DQN.EPSILON_DECAY,
#         gamma=config.DQN.GAMMA,
#         memory_size=config.DQN.MEMORY_SIZE
#     )
    
#     print(f"\n{Fore.GREEN}✅ Deux agents DQN créés avec succès!{Style.RESET_ALL}")
    
#     # Entraînement multi-agent
#     train_multi_agent(env=env, agent1=agent1, agent2=agent2)
    
#     # Test multi-agent
#     test_multi_agent(env=env, agent1=agent1, agent2=agent2, num_tests=20)


# if __name__ == "__main__":
#     main_multi_agent()
