#!/usr/bin/env python3

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from flappy_bird_env_v1 import FlappyBird_v1
import os
import numpy as np


models_dir = "models/FlappyBird_v1"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
print("Environment setup...")
env = FlappyBird_v1()
env.reset()

print("Creating model...")

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir,device="cpu")

TIMESTEPS = 10000
i = 0
print("Training...")
# callback = BiasedActionCallback(bias_steps_per_episode=150, flap_prob=0.05)
# for i in range(30):
while True:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="FlappyBird_v1")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    i += 1






    # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", callback=callback)

# class BiasedActionCallback(BaseCallback):
#     def __init__(self, bias_steps_per_episode=50, flap_prob=0.02):
#         super().__init__()
#         self.bias_steps_per_episode = bias_steps_per_episode
#         self.flap_prob = flap_prob
#         self.episode_step = 0
#
#     
#     def _on_step(self):
#         if self.episode_step < self.bias_steps_per_episode:
#             if np.random.random() > self.flap_prob:
#                 self.locals['actions'][0] = 0  # Force "nothing"
#             else:
#                 self.locals['actions'][0] = 1  # Force "nothing"
#         
#         self.episode_step += 1
#         
#         # Reset counter on episode end
#         dones = self.locals.get('dones')
#         if dones is not None and dones[0]:
#             self.episode_step = 0
#         
#         return True
