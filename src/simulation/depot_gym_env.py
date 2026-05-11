import gymnasium as gym
from gymnasium import spaces
import numpy as np
import duckdb

class GeoAwareMROEnv(gym.Env):
    # Updated path to the new DB location
    def __init__(self, db_path="data/db/geo_aware_mro.db"):
        super(GeoAwareMROEnv, self).__init__()
        self.db = duckdb.connect(db_path)
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        try:
            # Querying the Comtrade snapshots for trade volatility
            self.trade_series = self.db.execute(
                "SELECT trade_value_usd FROM comtrade_data ORDER BY period LIMIT 1000"
            ).df()['trade_value_usd'].values
            self.trade_series = (self.trade_series - self.trade_series.min()) / (self.trade_series.max() - self.trade_series.min())
        except Exception:
            self.trade_series = np.random.uniform(0.1, 1.0, 1000)

        self.state = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.trade_ptr = np.random.randint(0, len(self.trade_series) - 101)
        current_trade = self.trade_series[self.trade_ptr]
        self.state = np.array([0.5, 0.5, 0.5, current_trade], dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.steps += 1
        inventory, health, distance, _ = self.state
        trade_flow = self.trade_series[self.trade_ptr + self.steps]
        
        reward = 0.1
        if action == 1: # Resuscitate
            health = min(1.0, health + 0.2)
            reward += 0.5
        elif action == 2: # Restock
            # Global trade flow impacts restocking success
            restock_success = 0.3 * trade_flow 
            inventory = min(1.0, inventory + restock_success)
            reward += restock_success
            
        health -= 0.05
        self.state = np.array([inventory, health, distance, trade_flow], dtype=np.float32)
        
        terminated = self.steps >= 100
        if health <= 0:
            reward -= 2.0
            terminated = True
            
        return self.state, reward, terminated, False, {}
