import numpy as np
from stable_baselines3 import PPO
from src.simulation.depot_gym_env import GeoAwareMROEnv

# 1. Load the environment and trained model
env = GeoAwareMROEnv()
model = PPO.load("models/ppo_geo_mro_v1")

# 2. Define a "Struggling Depot" scenario
# [Inventory=0.2 (Low), Health=0.1 (Critical), Distance=0.5]
obs = np.array([0.2, 0.1, 0.5], dtype=np.float32)

# 3. Ask the agent for the best action
action, _states = model.predict(obs, deterministic=True)

# 4. Interpret the result
actions = {0: "Idle/Hold", 1: "Immediate Resuscitation", 2: "Restock Parts"}
print(f"\n--- Field Test ---")
print(f"Scenario: Inventory: 20%, Equipment Health: 10%")
print(f"Agent Recommendation: {actions[int(action)]}")
