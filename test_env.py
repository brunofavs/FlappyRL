#!/usr/bin/env python3

from stable_baselines3.common.env_checker import check_env
from flappy_bird_model import FlappyBird


env = FlappyBird()
# It will check your custom environment and output additional warnings if needed
check_env(env)
