#!/usr/bin/env python3
from flappy_bird_model import FlappyBird
import numpy as np

env = FlappyBird()
episodes = 10

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        random_action = np.random.choice([0, 1], p=[0.97, 0.03])
        obs, reward, done, truncated, info = env.step(random_action)

        # if random_action == 1:
        #     random_action = 0
        print("action",random_action)
        print('reward',reward)
        print('done',done)
