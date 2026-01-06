#!/usr/bin/env python3
from flappy_bird_env_v1 import FlappyBird_v1
import numpy as np
from pprint import pprint


env = FlappyBird_v1()
episodes = 1

for episode in range(episodes):
    done = False
    obs = env.reset()
    step_count = 0
    while not done:
        random_action = np.random.choice([0, 1], p=[0.97, 0.03])
        # random_action = 0
        obs, reward, done, truncated, info = env.step(random_action)

        # if random_action == 1:
        #     random_action = 0
        if step_count % 30 == 0:
            # print('info',info)
            print('\n\n')
            pprint(info)
            input("Press Enter to continue...")
            # print("action",random_action)
            # print('Reward per step',reward)
            # print('done',done)
            # print('obs',obs)
        step_count += 1

print(step_count)
# while True:
#     pass
