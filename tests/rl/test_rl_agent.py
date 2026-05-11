import pytest
import numpy as np
from src.rl.mro_env import MROInventoryEnv, GYM_AVAILABLE
from src.rl.rl_reorder_agent import PPOReorderAgent

@pytest.mark.skipif(not GYM_AVAILABLE, reason="gym not installed")
def test_env_resets():
    env = MROInventoryEnv(T=10)
    obs, info = env.reset(seed=0)
    assert obs.shape == (4,)
    assert ((obs >= 0.0) & (obs <= 1.0)).all()

@pytest.mark.skipif(not GYM_AVAILABLE, reason="gym not installed")
def test_env_terminates_at_T():
    T = 12
    env = MROInventoryEnv(T=T)
    env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.array([0.2], dtype=np.float32))
        done = term or trunc
        steps += 1
    assert steps == T

def test_agent_initializes():
    agent = PPOReorderAgent(env_params={"T": 10}, seed=42)
    assert agent.total_timesteps > 0

@pytest.mark.skipif(not GYM_AVAILABLE, reason="gym not installed")
def test_newsvendor_baseline_eval():
    agent = PPOReorderAgent(env_params={"T": 5, "mean_demand": 10, "q_max": 50}, seed=0)
    result = agent.evaluate_newsvendor_baseline(n_episodes=2)
    assert result.n_episodes == 2
    assert 0.0 <= result.mean_fill_rate <= 1.0
