"""Gymnasium wrapper around `DriveNavEnv` — the only module that imports gymnasium.

Keeping the RL dependency confined to one thin adapter is deliberate (and inherited from
`gsplat-rt`): the environment core in `nav_env.py` stays pure NumPy, so it runs and is fully
tested with no RL stack installed, and swapping stable-baselines3 for anything else touches
only this file.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .nav_env import DriveNavConfig, DriveNavEnv, SceneSource


class NavGymEnv(gym.Env):
    """Gymnasium view of `DriveNavEnv`. Continuous `[-1, 1]^2` actions, Box observations."""

    metadata = {"render_modes": []}

    def __init__(self, scenes: SceneSource, cfg: Optional[DriveNavConfig] = None):
        super().__init__()
        self.env = DriveNavEnv(scenes, cfg)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observations are normalised by construction but not hard-bounded — the goal-offset
        # channels exceed 1 when the goal is further away than the ray range, so the box is
        # left open rather than clipping values the policy legitimately needs to see.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.env.obs_dim,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        return self.env.reset(seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # `info` carries a VehicleState and a BEVGrid reference; strip them so vectorised
        # env wrappers can batch the dict without dragging whole grids through it.
        slim = {k: v for k, v in info.items() if k not in ("state",)}
        return obs, reward, terminated, truncated, slim
