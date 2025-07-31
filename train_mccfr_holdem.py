# train_mccfr_holdem.py

from open_spiel.python.algorithms import mccfr
import pyspiel
import time

# --------- Paramètres du jeu ---------
game_params = {
    "players": 2,
    "stack": 1000,
    "blind": 50,
    "max_raises": 4,
    "betting": "no_limit"
}

game_name = "texas_holdem_no_limit"
game = pyspiel.load_game(game_name, game_params)

# --------- Initialisation du solveur MCCFR ---------
solver = mccfr.MCCFRSolver(game)

# --------- Paramètres d'entraînement ---------
num_iterations = 5000
log_interval = 100

print(f"Training MCCFR on {game_name}...")
start_time = time.time()

for i in range(1, num_iterations + 1):
    solver.iteration()
    
    if i % log_interval == 0 or i == 1:
        exploit = solver.exploitability()
        print(f"Iteration {i}/{num_iterations} - Exploitability: {exploit:.4f}")

end_time = time.time()
duration = end_time - start_time
print(f"\nTraining completed in {duration:.2f} seconds.")

# --------- Sauvegarder la stratégie moyenne ---------
policy = solver.average_policy()
policy_file = "average_strategy_policy.txt"

with open(policy_file, "w") as f:
    for state_key in policy.policy.keys():
        actions = policy.policy[state_key]
        f.write(f"{state_key}:\n")
        for action_str, prob in actions.items():
            f.write(f"  {action_str}: {prob:.4f}\n")
        f.write("\n")

print(f"\nStratégie moyenne sauvegardée dans : {policy_file}")
