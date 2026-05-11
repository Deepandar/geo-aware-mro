import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False

if GYM_AVAILABLE:
    class MROInventoryEnv(gym.Env):
        def __init__(self, T=52, q_max=50.0, max_inv=200.0, mean_demand=10.0, std_demand=3.0, 
                     unit_cost=100.0, stockout_cost=1000.0, holding_rate=0.20, 
                     order_fixed_cost=50.0, rul_threshold=20.0, rul_max=200.0, 
                     geo_risk_schedule=None, seed=42):
            super().__init__()
            self.T = T
            self.q_max = q_max
            self.max_inv = max_inv
            self.mean_demand = mean_demand
            self.std_demand = std_demand
            self.unit_cost = unit_cost
            self.stockout_cost = stockout_cost
            self.holding_rate = holding_rate
            self.order_fixed_cost = order_fixed_cost
            self.rul_threshold = rul_threshold
            self.rul_max = rul_max
            self.geo_risk_schedule = geo_risk_schedule or {}
            self._seed = seed
            self._cost_scale = (q_max * unit_cost * holding_rate + q_max * stockout_cost + order_fixed_cost)
            self.observation_space = spaces.Box(low=np.zeros(4, dtype=np.float32), high=np.ones(4, dtype=np.float32), dtype=np.float32)
            self.action_space = spaces.Box(low=np.zeros(1, dtype=np.float32), high=np.ones(1, dtype=np.float32), dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            rng_seed = seed if seed is not None else self._seed
            self._rng = np.random.default_rng(rng_seed)
            self._inventory = float(self._rng.uniform(self.mean_demand * 0.5, self.mean_demand * 2.5))
            self._period = 0
            self._geo_risk = float(self._rng.uniform(0.0, 0.4))
            self._rul = float(self._rng.uniform(self.rul_threshold * 0.5, self.rul_max))
            self._total_reward = 0.0
            self._stockout_events = 0
            self._total_filled = 0.0
            self._total_demand = 0.0
            return self._get_obs(), {}

        def step(self, action):
            order_fraction = float(np.clip(action[0], 0.0, 1.0))
            order_qty = order_fraction * self.q_max
            fixed_cost = self.order_fixed_cost if order_qty > 0.1 else 0.0
            self._inventory += order_qty
            demand = max(float(self._rng.normal(self.mean_demand, self.std_demand)), 0.0)
            self._total_demand += demand
            filled = min(self._inventory, demand)
            shortfall = demand - filled
            self._total_filled += filled
            self._inventory = min(max(self._inventory - demand, 0.0), self.max_inv)
            if shortfall > 0:
                self._stockout_events += 1
            holding_cost = (self._inventory * self.unit_cost * self.holding_rate)
            stockout_cost = shortfall * self.stockout_cost
            reward = -(holding_cost + stockout_cost + fixed_cost) / max(self._cost_scale, 1.0)
            self._total_reward += reward
            self._period += 1
            if self._period in self.geo_risk_schedule:
                self._geo_risk = float(self.geo_risk_schedule[self._period])
            else:
                self._geo_risk = float(np.clip(self._geo_risk + self._rng.normal(0, 0.02), 0.0, 1.0))
            self._rul = max(self._rul - self._rng.uniform(0.5, 2.0), 0.0)
            terminated = self._period >= self.T
            return self._get_obs(), reward, terminated, False, {"fill_rate": self._total_filled / max(self._total_demand, 1)}

        def _get_obs(self):
            return np.array([self._inventory / self.max_inv, self._period / self.T, self._geo_risk, self._rul / self.rul_max], dtype=np.float32)

        def episode_summary(self):
            return {"total_reward": round(self._total_reward, 4), "fill_rate": round(self._total_filled / max(self._total_demand, 1), 4), "stockout_events": self._stockout_events}
else:
    class MROInventoryEnv:
        def __init__(self, **kwargs): raise ImportError("pip install gymnasium")
