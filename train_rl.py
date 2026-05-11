import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from stable_baselines3 import PPO
from src.rl.mro_env import MROInventoryEnv
from src.rl.rl_reorder_agent import PPOReorderAgent

# 1. Initialize Environment
env_params = {
    "T": 52,
    "mean_demand": 12.0,
    "std_demand": 4.0,
    "unit_cost": 150.0,
    "holding_rate": 0.15,
    "stockout_cost": 1200.0
}
env = MROInventoryEnv(**env_params)

# 2. Train the PPO Model
print("\n--- Starting Training (100,000 steps) ---")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_mro_tensorboard/")
model.learn(total_timesteps=100000)

# Save the model
if not os.path.exists("models"):
    os.makedirs("models")
model.save("models/ppo_mro_agent_v1")
print("Model saved to models/ppo_mro_agent_v1")

# 3. Comparative Evaluation
print("\n--- Running Evaluation ---")
agent = PPOReorderAgent(env_params=env_params)
baseline_results = agent.evaluate_newsvendor_baseline(n_episodes=100, newsvendor_q_star=15.0)

rl_rates = []
for i in range(100):
    obs, _ = env.reset(seed=42+i)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    rl_rates.append(info["fill_rate"])

print(f"\n--- Week 30 Final Results ---")
print(f"Newsvendor Baseline Fill Rate: {baseline_results.mean_fill_rate:.2%}")
print(f"PPO RL Agent Fill Rate:        {np.mean(rl_rates):.2%}")
print("------------------------------")
