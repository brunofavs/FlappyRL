#!/usr/bin/env python3

import numpy as np
import gymnasium as gym

class FlappyBird(gym.Env):

    def __init__(self):

        # Initialize positions - will be set randomly in reset()

        # Define what the agent can observe, Placeholder
        self.observation_space = gym.spaces.Discrete(8)

        # Define what actions are available (0: do nothing, 1: flap)
        self.action_space = gym.spaces.Discrete(2)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observations 
        """
        return {}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return { }


    # https://chatgpt.com/s/t_694b5f4d79c081918371fd8e67d84dcb
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        reward = 0.0
        terminated = False
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
