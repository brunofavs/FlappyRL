#!/usr/bin/env python3

from stable_baselines3.common.env_checker import check_env
from flappy_bird_env_v1 import FlappyBird_v1


env = FlappyBird_v1()
# It will check your custom environment and output additional warnings if needed
check_env(env)
