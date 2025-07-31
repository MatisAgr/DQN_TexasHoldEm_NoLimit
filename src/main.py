# lancer ce fichier

import os
from config import config

# Configuration TensorFlow avec les paramètres du config
# os.environ["OMP_NUM_THREADS"] = str(config.TENSORFLOW_THREADS)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduit les logs TensorFlow

import tensorflow as tf
# utiliser tous les threads disponibles (n'a pas l'air de marcher)
# tf.config.threading.set_intra_op_parallelism_threads(config.TENSORFLOW_THREADS)
# tf.config.threading.set_inter_op_parallelism_threads(config.TENSORFLOW_THREADS)

import numpy as np
import time
from colorama import Fore, Style

from poker_game import TreysPokerEnv, PokerAction
from console_display import PokerConsole
from dqn_agent import DQNAgent


def train_agent(env: TreysPokerEnv, agent: DQNAgent, episodes: int = config.TRAINING.EPISODES, 
                show_game_every: int = config.TRAINING.SHOW_GAME_EVERY, show_opponent_cards: bool = config.TRAINING.SHOW_OPPONENT_CARDS,
                train_frequency: int = config.TRAINING.TRAIN_FREQUENCY, target_update_frequency: int = config.TRAINING.TARGET_UPDATE_FREQUENCY) -> None:
    
    reward_history = []
    win_history = []
    loss_history = []
    step_count = 0
    
    # Affichage initial
    PokerConsole.clear_screen()
    PokerConsole.print_header("ENTRAINEMENT DQN POKER AVEC TREYS")
    print(f"{Fore.YELLOW}Entrainement de l'agent DQN pour {episodes} episodes...{Style.RESET_ALL}")

    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps_in_episode = 0
        episode_loss = 0
        
        # Affichage de la partie si c'est le bon intervalle
        show_this_game = (episode % show_game_every == 0 and episode > 0)
        
        if show_this_game:
            PokerConsole.clear_screen()
            print(f"{Fore.YELLOW}Demonstration - Episode {episode}{Style.RESET_ALL}")
            PokerConsole.render_game(env, show_opponent_cards)
            time.sleep(1)
        
        while True:
            # Action du joueur avec epsilon-greedy
            action = agent.act(state, training=True)
            
            next_state, reward, done, info = env.step(action)
            
            # Affichage de l'etat apres l'action si demonstration
            if show_this_game:
                PokerConsole.render_game(env, show_opponent_cards)
                if not done:
                    time.sleep(1)
            
            # Stockage de la transition
            agent.store_transition(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            steps_in_episode += 1
            step_count += 1
            
            # Entrainement periodique
            if step_count % train_frequency == 0:
                loss = agent.train_step()
                if loss is not None:
                    episode_loss += loss
            
            if done:
                if show_this_game:
                    print(f"\n{Fore.CYAN}Fin de la partie - Recompense totale: {total_reward:.1f}{Style.RESET_ALL}")
                    time.sleep(2)
                break
        
        # Mise a jour de l'epsilon
        agent.update_epsilon()
        
        # Mise a jour du target network
        if episode % target_update_frequency == 0:
            agent.update_target_model()
        
        # Statistiques
        reward_history.append(total_reward)
        win_history.append(1 if total_reward > 0 else 0)
        loss_history.append(episode_loss / max(1, steps_in_episode))
        
        # Affichage des resultats periodique
        if episode % config.TRAINING.PROGRESS_EVERY == 0:
            recent_wins = sum(win_history[-100:]) if len(win_history) >= 100 else sum(win_history)
            recent_episodes = min(100, len(win_history))
            win_rate = recent_wins / recent_episodes * 100
            avg_reward = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
            avg_loss = np.mean(loss_history[-100:]) if len(loss_history) >= 100 else np.mean(loss_history)
            
            PokerConsole.print_training_progress(episode, total_reward, agent.epsilon, 
                                               win_rate, avg_reward, avg_loss)
        
        # Sauvegarde periodique
        if episode % config.TRAINING.SAVE_EVERY == 0 and episode > 0:
            save_path = config.PATHS.EPISODE_MODEL_TEMPLATE.format(episode)
            agent.save_model(save_path)
            print(f"{Fore.GREEN}Modele sauvegarde a l'episode {episode}{Style.RESET_ALL}")
    
    # Resultats finaux
    final_win_rate = sum(win_history[-200:]) / min(200, len(win_history)) * 100
    final_avg_reward = np.mean(reward_history[-200:])
    final_avg_loss = np.mean(loss_history[-200:]) if loss_history else 0
    
    PokerConsole.print_final_results(episodes, final_win_rate, final_avg_reward, 
                                   final_avg_loss, agent.get_memory_size())
    
    # Sauvegarde finale
    agent.save_model(config.PATHS.FINAL_MODEL)
    print(f"{Fore.GREEN}Modele final sauvegarde!{Style.RESET_ALL}")


def test_agent(env: TreysPokerEnv, agent: DQNAgent, num_tests: int = 20) -> None:
    """
    Teste l'agent entraine sur plusieurs parties
    """
    PokerConsole.print_header("TEST DU MODELE ENTRAINE")
    print(f"\n{Fore.YELLOW}Test du modele sur {num_tests} parties (mode exploitation pur)...{Style.RESET_ALL}")
    
    test_wins = 0
    test_rewards = []
    
    for test_episode in range(num_tests):
        state = env.reset()
        episode_reward = 0
        
        # Affichage pour les 3 premieres parties de test
        show_test = test_episode < 3
        
        if show_test:
            print(f"\n{Fore.CYAN}Partie de test #{test_episode + 1}{Style.RESET_ALL}")
            PokerConsole.render_game(env, show_opponent_cards=True)
            time.sleep(1)
        
        while True:
            # Mode test : pas d'exploration
            action = agent.act(state, training=False)
            
            if show_test:
                action_enum = PokerAction(action)
                PokerConsole.print_action("Agent IA", action_enum)
                q_values = agent.get_q_values(state)
                print(f"Q-values: {[f'{q:.2f}' for q in q_values]}")
                time.sleep(1)
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if show_test:
                PokerConsole.render_game(env, show_opponent_cards=True)
                if not done:
                    time.sleep(1)
            
            if done:
                if show_test:
                    print(f"\n{Fore.CYAN}Fin de la partie de test - Recompense: {episode_reward:.1f}{Style.RESET_ALL}")
                    time.sleep(2)
                break
        
        test_rewards.append(episode_reward)
        if episode_reward > 0:
            test_wins += 1
        
        # Affichage resume pour chaque test
        result_color = Fore.GREEN if episode_reward > 0 else Fore.RED
        result_text = "Victoire" if episode_reward > 0 else "Defaite"
        
        print(f"Test {test_episode + 1:2d}: {result_color}Recompense = {episode_reward:6.1f}{Style.RESET_ALL} "
              f"- {result_text}")
    
    # Affichage des resultats
    PokerConsole.print_test_results(test_wins, num_tests, test_rewards)
    
    # Message final selon les performances
    test_win_rate = test_wins / num_tests * 100
    print(f"\n{Fore.CYAN}Votre agent DQN pour le poker avec Treys est pret!")
    print(f"L'evaluation des mains est maintenant precise grace a la bibliotheque Treys.{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Ameliorations possibles: bluff plus sophistique, position, stack management...{Style.RESET_ALL}")


def main():
    """Fonction principale du programme"""
    # Utilisation des paramètres de configuration
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)
    
    # Création des répertoires nécessaires
    os.makedirs(config.PATHS.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.PATHS.LOGS_DIR, exist_ok=True)
    
    # Initialisation
    env = TreysPokerEnv()
    
    print(f"{Fore.CYAN}Environnement Poker initialise avec Treys")
    print(f"Dimension de l'etat: {env.state_size}")
    print(f"Nombre d'actions: {env.num_actions}")
    print(f"Actions disponibles: {[action.name for action in PokerAction]}{Style.RESET_ALL}")
    
    # Creation de l'agent DQN avec les paramètres de configuration
    agent = DQNAgent(
        state_size=env.state_size,
        num_actions=env.num_actions,
        learning_rate=config.DQN.LEARNING_RATE,
        epsilon=config.DQN.EPSILON_START,
        epsilon_min=config.DQN.EPSILON_MIN,
        epsilon_decay=config.DQN.EPSILON_DECAY,
        gamma=config.DQN.GAMMA,
        memory_size=config.DQN.MEMORY_SIZE
    )
    
    # Affichage de l'architecture
    agent.get_model_summary()
    
    # Entrainement avec les paramètres de configuration
    train_agent(env=env, agent=agent)
    
    # Test
    test_agent(env=env, agent=agent, num_tests=20)


if __name__ == "__main__":
    main()
