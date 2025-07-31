import rlcard
from rlcard.agents import RandomAgent

env = rlcard.make('no-limit-holdem')
env.set_agents([RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)])

print(env.num_actions) # 5
print(env.num_players) # 2
print(env.state_shape) # [[54], [54]]
print(env.action_shape) # [None, None]

trajectories, payoffs = env.run()