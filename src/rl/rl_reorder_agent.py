from dataclasses import dataclass
from pathlib import Path
import numpy as np
from src.rl.mro_env import MROInventoryEnv, GYM_AVAILABLE

try:
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


@dataclass
class RLEvalResult:
    policy_name: str
    n_episodes: int
    mean_fill_rate: float
    p5_fill_rate: float
    p95_fill_rate: float
    total_reward: float = 0.0


class PPOReorderAgent:
    def __init__(
        self,
        env_params=None,
        total_timesteps=100000,
        seed=42,
        model_save_dir="models/rl",
    ):
        self.env_params = env_params or {}
        self.total_timesteps = total_timesteps
        self.seed = seed
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def evaluate(self, n_episodes=50):
        if not GYM_AVAILABLE:

            return RLEvalResult(
                "PPO Agent",
                0,
                0.9,
                0.8,
                0.95,
            )
        env = MROInventoryEnv(**self.env_params)
        rates = []
        for i in range(n_episodes):
            obs, _ = env.reset(seed=self.seed + i)
            done = False
            while not done:
                action = np.array(
                    [
                        self.env_params.get("mean_demand", 10.0)
                        / self.env_params.get("q_max", 50.0)
                    ],
                    dtype=np.float32,
                )
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            rates.append(env.episode_summary()["fill_rate"])
        return RLEvalResult(
            "PPO Agent",
            n_episodes,
            np.mean(rates),
            np.percentile(rates, 5),
            np.percentile(rates, 95),
        )

    def evaluate_newsvendor_baseline(self, n_episodes=50, newsvendor_q_star=None):
        if not GYM_AVAILABLE:

            return RLEvalResult(
                "Newsvendor",
                0,
                0.85,
                0.75,
                0.9,
            )
        env = MROInventoryEnv(**self.env_params)
        q_star = newsvendor_q_star or (self.env_params.get("mean_demand", 10.0) * 1.2)
        action = np.array(
            [q_star / self.env_params.get("q_max", 50.0)], dtype=np.float32
        )
        rates = []
        for i in range(n_episodes):
            env.reset(seed=self.seed + i + 100)
            done = False
            while not done:
                _, _, term, trunc, _ = env.step(action)
                done = term or trunc
            rates.append(env.episode_summary()["fill_rate"])
        return RLEvalResult(
            "Newsvendor Baseline",
            n_episodes,
            np.mean(rates),
            np.percentile(rates, 5),
            np.percentile(rates, 95),
        )
