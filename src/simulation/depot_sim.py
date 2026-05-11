import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GeoAwareMROEnv(gym.Env):
    """
    Custom Environment for MRO Base Resuscitation and Supply Chain Optimization.
    """
    def __init__(self):
        super(GeoAwareMROEnv, self).__init__()
        
        # Action Space: 0 = No Action, 1 = Resuscitate Small Arms, 2 = Restock Parts
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: [Inventory_Level, Equipment_Health, Distance_to_HQ]
        # Using a Box for continuous values normalized between 0 and 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        self.state = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize state: Randomly starting with high health/inventory for testing
        self.state = np.array([0.8, 0.9, 0.5], dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.steps += 1
        
        # Unpack state
        inventory, health, distance = self.state
        
        # Logic: Actions affect health and inventory
        reward = 0
        if action == 1:  # Resuscitate
            health = min(1.0, health + 0.1)
            reward += 1.0
        elif action == 2: # Restock
            inventory = min(1.0, inventory + 0.2)
            reward += 0.5
        
        # Decay health over time (simulating usage)
        health -= 0.05
        
        # Update state
        self.state = np.array([inventory, health, distance], dtype=np.float32)
        
        # Terminate after 100 steps
        terminated = self.steps >= 100
        truncated = False
        
        # Penalty for low health (Down-time)
        if health < 0.2:
            reward -= 2.0
            
        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"Step: {self.steps} | State: {self.state}")