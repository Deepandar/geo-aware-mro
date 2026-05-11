from stable_baselines3 import PPO
from src.simulation.depot_gym_env import GeoAwareMROEnv

# 1. Initialize the Environment
env = GeoAwareMROEnv()

# 2. Define the PPO Model (Using MlpPolicy for structured data)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo_mro/")

# 3. Start Learning
print("--- Starting MRO Resuscitation Training ---")
model.learn(total_timesteps=10000)

# 4. Save the Model
model.save("models/ppo_geo_mro_v1")
print("--- Training Complete. Model saved to models/ ---")
