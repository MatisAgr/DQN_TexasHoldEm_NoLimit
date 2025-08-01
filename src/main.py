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
    
    # on garde trace des metriques d'entrainement
    reward_history = []
    win_history = []
    loss_history = []
    temperature_history = []  # nouvelle metrique pour suivre la temperature
    step_count = 0
    
    # affichage initial pour suivre le progres
    PokerConsole.clear_screen()
    PokerConsole.print_header("entrainement dqn poker avec treys - exploration softmax")
    print(f"{Fore.YELLOW}entrainement de l'agent dqn pour {episodes} episodes avec exploration par temperature...{Style.RESET_ALL}")

    
    for episode in range(episodes):
        # reset de l'environnement pour un nouvel episode
        state = env.reset()
        total_reward = 0
        steps_in_episode = 0
        episode_loss = 0
        episode_q_values = []
        
        # on affiche certaines parties pour voir comment ca se passe
        show_this_game = (episode % show_game_every == 0 and episode > 0)
        
        if show_this_game:
            PokerConsole.clear_screen()
            print(f"{Fore.YELLOW}demonstration - episode {episode}{Style.RESET_ALL}")
            PokerConsole.render_game(env, show_opponent_cards)
            time.sleep(1)
        
        while True:
            # l'agent choisit une action avec exploration softmax basee sur la temperature
            action = agent.act(state, training=True, temperature=agent.temperature)
            
            # on garde les q-values pour les stats
            q_values = agent.get_q_values(state)
            episode_q_values.append(np.max(q_values))
            
            next_state, reward, done, info = env.step(action)
            
            # affichage de l'etat apres l'action si demonstration
            if show_this_game:
                PokerConsole.render_game(env, show_opponent_cards)
                if not done:
                    time.sleep(1)
            
            # stockage de la transition
            agent.store_transition(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            steps_in_episode += 1
            step_count += 1
            
            # entrainement periodique (sans callbacks pour eviter les problemes)
            if step_count % train_frequency == 0:
                loss = agent.train_step(use_callbacks=False)
                if loss is not None:
                    episode_loss += loss
            
            if done:
                if show_this_game:
                    print(f"\n{Fore.CYAN}fin de la partie - recompense totale: {total_reward:.1f}{Style.RESET_ALL}")
                    time.sleep(2)
                break
        
        # mise a jour de la temperature (remplace epsilon)
        agent.update_temperature()
        
        # mise a jour du target network
        if episode % target_update_frequency == 0:
            agent.update_target_model()
        
        # enregistrer les statistiques d'episode
        avg_q_value = np.mean(episode_q_values) if episode_q_values else 0
        avg_episode_loss = episode_loss / max(1, steps_in_episode)
        agent.record_episode_stats(total_reward, steps_in_episode, avg_q_value)
        
        # enregistrer dans tensorboard
        is_win = total_reward > 0
        agent.log_to_tensorboard(episode, total_reward, avg_episode_loss, steps_in_episode, is_win)
        
        # statistiques
        reward_history.append(total_reward)
        win_history.append(1 if total_reward > 0 else 0)
        loss_history.append(avg_episode_loss)
        temperature_history.append(agent.temperature)  # suivre l'evolution de la temperature
        
        # affichage des statistiques en temps reel avec temperature au lieu d'epsilon
        if episode % 10 == 0 or episode < 10:  # afficher plus souvent au debut
            recent_wins = sum(win_history[-50:]) if len(win_history) >= 50 else sum(win_history)
            recent_episodes = min(50, len(win_history))
            win_rate = recent_wins / recent_episodes * 100 if recent_episodes > 0 else 0
            
            avg_reward_recent = np.mean(reward_history[-50:]) if len(reward_history) >= 50 else np.mean(reward_history)
            avg_loss_recent = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else np.mean(loss_history)
            
            print(f"\n\n\t\t\t----- episode: {episode:4d} | reward: {total_reward:8.1f} | temperature: {agent.temperature:.4f} -----")
            print(f"\t\t\t----- win rate (50): {win_rate:5.1f}% | avg reward: {avg_reward_recent:6.1f} | avg loss: {avg_loss_recent:.4f} -----")
            print(f"\t\t\t----- memory: {agent.get_memory_size():5d} | steps: {steps_in_episode:3d} | q-avg: {avg_q_value:.2f} -----\n\n")
        
        # affichage des resultats periodique detaille
        if episode % config.TRAINING.PROGRESS_EVERY == 0 and episode > 0:
            agent.print_training_stats(episode, window=100)
            recent_wins = sum(win_history[-100:]) if len(win_history) >= 100 else sum(win_history)
            recent_episodes = min(100, len(win_history))
            win_rate = recent_wins / recent_episodes * 100
            avg_reward = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
            avg_loss = np.mean(loss_history[-100:]) if len(loss_history) >= 100 else np.mean(loss_history)
            
            # affichage modifie pour montrer la temperature au lieu d'epsilon
            print(f"\n{Fore.MAGENTA}=== progres episode {episode} ==={Style.RESET_ALL}")
            print(f"{Fore.CYAN}temperature actuelle: {agent.temperature:.4f} | win rate (100): {win_rate:.1f}%{Style.RESET_ALL}")
            print(f"{Fore.CYAN}avg reward (100): {avg_reward:.1f} | avg loss (100): {avg_loss:.4f}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}memoire utilisee: {agent.get_memory_size()} transitions{Style.RESET_ALL}")
        
        # sauvegarde periodique
        if episode % config.TRAINING.SAVE_EVERY == 0 and episode > 0:
            save_path = config.PATHS.EPISODE_MODEL_TEMPLATE.format(episode)
            agent.save_model(save_path)
            print(f"{Fore.GREEN}modele sauvegarde a l'episode {episode}{Style.RESET_ALL}")
    
    # resultats finaux
    final_win_rate = sum(win_history[-200:]) / min(200, len(win_history)) * 100
    final_avg_reward = np.mean(reward_history[-200:])
    final_avg_loss = np.mean(loss_history[-200:]) if loss_history else 0
    final_temperature = agent.temperature
    
    print(f"\n{Fore.MAGENTA}=== resultats finaux ==={Style.RESET_ALL}")
    print(f"{Fore.GREEN}episodes total: {episodes}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}win rate final (200 derniers): {final_win_rate:.1f}%{Style.RESET_ALL}")
    print(f"{Fore.GREEN}reward moyen final: {final_avg_reward:.1f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}loss moyenne finale: {final_avg_loss:.4f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}temperature finale: {final_temperature:.4f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}memoire finale: {agent.get_memory_size()} transitions{Style.RESET_ALL}")
    
    # sauvegarde finale
    agent.save_model(config.PATHS.FINAL_MODEL)
    print(f"{Fore.GREEN}modele final sauvegarde!{Style.RESET_ALL}")
    
    # fermer tensorboard
    agent.close_tensorboard()
    print(f"{Fore.CYAN}logs tensorboard sauvegardes dans: {agent.tensorboard_log_dir}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}pour voir les logs tensorboard, executez: tensorboard --logdir={agent.tensorboard_log_dir}{Style.RESET_ALL}")


def main():
    """fonction principale du programme"""
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}dqn poker avec exploration softmax")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    # TODO: choix du mode d'entrainement
    # if config.TRAINING.MULTI_AGENT_MODE:
    #     from multi_agent_training import main_multi_agent
    #     main_multi_agent()
    #     return
        
    # utilisation des parametres de configuration
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)
    
    # creation des repertoires necessaires
    os.makedirs(config.PATHS.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.PATHS.LOGS_DIR, exist_ok=True)
    
    # initialisation
    env = TreysPokerEnv()
        
    # creation de l'agent dqn avec les parametres de configuration
    # note: on garde les anciens parametres epsilon dans config mais on les ignore
    agent = DQNAgent(
        state_size=env.state_size,
        num_actions=env.num_actions,
        learning_rate=config.DQN.LEARNING_RATE,
        gamma=config.DQN.GAMMA,
        memory_size=config.DQN.MEMORY_SIZE,
        # parametres specifiques pour softmax
        temperature_start=2.0,  # temperature initiale haute pour plus d'exploration
        temperature_min=0.1,    # temperature minimale
        temperature_decay=0.995 # decay plus lent que epsilon
    )
    
    # affichage de l'architecture
    agent.get_model_summary()
    print(f"{Fore.YELLOW}exploration par softmax avec temperature initiale: {agent.temperature:.2f}{Style.RESET_ALL}")
        
    # entrainement seulement
    train_agent(env=env, agent=agent)

if __name__ == "__main__":
    main()