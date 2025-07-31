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
        episode_q_values = []
        
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
            
            # Enregistrer les Q-values pour les statistiques
            q_values = agent.get_q_values(state)
            episode_q_values.append(np.max(q_values))
            
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
            
            # Entrainement periodique (sans callbacks pour éviter les problèmes)
            if step_count % train_frequency == 0:
                loss = agent.train_step(use_callbacks=False)
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
        
        # Enregistrer les statistiques d'épisode
        avg_q_value = np.mean(episode_q_values) if episode_q_values else 0
        avg_episode_loss = episode_loss / max(1, steps_in_episode)
        agent.record_episode_stats(total_reward, steps_in_episode, avg_q_value)
        
        # Enregistrer dans TensorBoard
        is_win = total_reward > 0
        agent.log_to_tensorboard(episode, total_reward, avg_episode_loss, steps_in_episode, is_win)
        
        # Statistiques
        reward_history.append(total_reward)
        win_history.append(1 if total_reward > 0 else 0)
        loss_history.append(avg_episode_loss)
        
        # Affichage des statistiques en temps réel comme dans votre exemple
        if episode % 10 == 0 or episode < 10:  # Afficher plus souvent au début
            recent_wins = sum(win_history[-50:]) if len(win_history) >= 50 else sum(win_history)
            recent_episodes = min(50, len(win_history))
            win_rate = recent_wins / recent_episodes * 100 if recent_episodes > 0 else 0
            
            avg_reward_recent = np.mean(reward_history[-50:]) if len(reward_history) >= 50 else np.mean(reward_history)
            avg_loss_recent = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else np.mean(loss_history)
            
            print(f"\n\n\t\t\t----- Episode: {episode:4d} | Reward: {total_reward:8.1f} | Epsilon: {agent.epsilon:.4f} -----")
            print(f"\t\t\t----- Win Rate (50): {win_rate:5.1f}% | Avg Reward: {avg_reward_recent:6.1f} | Avg Loss: {avg_loss_recent:.4f} -----")
            print(f"\t\t\t----- Memory: {agent.get_memory_size():5d} | Steps: {steps_in_episode:3d} | Q-avg: {avg_q_value:.2f} -----\n\n")
        
        # Affichage des resultats periodique détaillé
        if episode % config.TRAINING.PROGRESS_EVERY == 0 and episode > 0:
            agent.print_training_stats(episode, window=100)
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
    
    # Fermer TensorBoard
    agent.close_tensorboard()
    print(f"{Fore.CYAN}Logs TensorBoard sauvegardes dans: {agent.tensorboard_log_dir}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Pour voir les logs TensorBoard, executez: tensorboard --logdir={agent.tensorboard_log_dir}{Style.RESET_ALL}")


def main():
    """Fonction principale du programme"""
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}DQN")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    # Choix du mode d'entraînement
    if config.TRAINING.MULTI_AGENT_MODE:
        from multi_agent_training import main_multi_agent
        main_multi_agent()
        return
        
    # Utilisation des paramètres de configuration
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)
    
    # Création des répertoires nécessaires
    os.makedirs(config.PATHS.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.PATHS.LOGS_DIR, exist_ok=True)
    
    # Initialisation
    env = TreysPokerEnv()
        
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
        
    # Entrainement seulement
    train_agent(env=env, agent=agent)

if __name__ == "__main__":
    main()
