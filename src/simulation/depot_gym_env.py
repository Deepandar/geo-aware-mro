import gymnasium as gym
from gymnasium import spaces
import numpy as np
import duckdb


class GeoAwareMROEnv(gym.Env):
    # Updated path to the new DB location
    def __init__(
        self,
        sku_df=None,
        seed=None,
        fast_mode=False,
        db_path="data/mro.duckdb",
        **kwargs
    ):
        self.fast_mode = fast_mode
        self.seed = seed
        super(GeoAwareMROEnv, self).__init__()
        try:
            self.db = duckdb.connect(db_path)
        except Exception:
            self.db = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        try:

            if self.db is not None:

                self.trade_series = self.db.execute("""
                        SELECT trade_value_usd
                        FROM comtrade_data
                        ORDER BY period
                        LIMIT 1000
                        """).df()["trade_value_usd"].values

                self.trade_series = (self.trade_series - self.trade_series.min()) / (
                    self.trade_series.max() - self.trade_series.min()
                )

            else:

                raise RuntimeError("DuckDB unavailable")

        except Exception:

            self.trade_series = np.random.uniform(
                0.1,
                1.0,
                1000,
            )

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
        # ... [Your logic for updating state, health, and inventory] ...

        # --- The Reward Logic we added ---
        MAINTENANCE_COST = -0.7
        INVENTORY_COST = -0.4
        reward = 0.1

        terminated = False
        truncated = False  # Usually False unless you have a time limit

        if action == 1:  # Resuscitation
            if self.state[1] < 0.7:
                reward += 0.5
            reward += MAINTENANCE_COST
        elif action == 2:  # Restock
            reward += 0.3
            reward += INVENTORY_COST

        if self.state[1] <= 0:
            reward = -2.0
            terminated = True

        # --- THE MISSING PART ---
        # Ensure 'self.state' is updated before this
        observation = self.state
        info = {}  # Can be empty, but must be a dict

        return observation, reward, terminated, truncated, info
