#!/usr/bin/env python3

import gymnasium as gym
from stable_baselines3 import PPO
from flappy_bird_env_v0 import FlappyBird

models_dir = "../models/FlappyBird_v0"

env = FlappyBird()
env.reset()

model_path = f"{models_dir}/190000.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, truncated, info = env.step(action)
        print(rewards)
