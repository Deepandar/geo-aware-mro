import numpy as np
from stable_baselines3 import PPO
from src.simulation.depot_gym_env import GeoAwareMROEnv

# Load environment and the fresh 4D model
env = GeoAwareMROEnv()
model = PPO.load("models/ppo_geo_mro_v1")

# Test Scenario: Critical Health AND Restricted Trade (Global Shortage)
# [Inventory=0.3, Health=0.15, Distance=0.5, Trade_Flow=0.1 (LOW)]
obs_shortage = np.array([0.3, 0.15, 0.5, 0.1], dtype=np.float32)

# Test Scenario: Critical Health AND Healthy Trade (Open Supply Lines)
# [Inventory=0.3, Health=0.15, Distance=0.5, Trade_Flow=0.9 (HIGH)]
obs_open = np.array([0.3, 0.15, 0.5, 0.9], dtype=np.float32)

actions = {0: "Idle/Hold", 1: "Immediate Resuscitation", 2: "Restock Parts"}

print("\n--- Supply Chain Stress Test ---")
action_s, _ = model.predict(obs_shortage, deterministic=True)
print(f"Scenario 1 (Supply Shortage): Recommendation -> {actions[int(action_s)]}")

action_o, _ = model.predict(obs_open, deterministic=True)
print(f"Scenario 2 (Open Supply):   Recommendation -> {actions[int(action_o)]}")
