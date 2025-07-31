from dqn_agent import DQNAgent
from en_poker import PokerEnv
import numpy as np
import matplotlib.pyplot as plt

episode = 30

def train():
    env = PokerEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    rewards = []
    max_steps = 200  # Sécurité pour éviter les boucles infinies
    list_action = ['Fold', 'Check', 'Call', 'Raise Half Pot', 'Raise Full Pot', 'All In']
    total_reward = 0
    for ep in range(episode):
        state, legal_actions = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = agent.act(state, legal_actions)
            next_state, reward, done, next_legal = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state, legal_actions = next_state, next_legal
            total_reward += reward
            print(f"Episode {ep}, Tour de la partie actuelle: {steps}, Type de : {list_action[action]}, Reward: {reward}, Total Reward: {total_reward}")
            agent.replay()
            steps += 1
        rewards.append(total_reward)
        if ep % 100 == 0:
            print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
    plt.title("Moving average reward (window=100)")
    plt.show()

if __name__ == "__main__":
    train()