#!/usr/bin/env python3

import gymnasium as gym
from stable_baselines3 import PPO
from flappy_bird_env_v1 import FlappyBird_v1
from pprint import pprint

models_dir = "models/FlappyBird_v1"

env = FlappyBird_v1()
env.reset()

model_path = f"{models_dir}/230000.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    step_count = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, truncated, info = env.step(action)

        if step_count % 30 == 0:
            print('\n\n')
            pprint(info)
            print('\n')
            print("Reward:", rewards)
            # input("Press Enter to continue...")

        step_count += 1
