import gym
import pybulletgym
import numpy as np
import torch as th
import argparse
# from stable_baselines3 import PPO
# from stable_baselines3.ppo import ppo as PPO
from stable_baselines3.ppo.ppo import PPO

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str)
parser.add_argument('--n-seeds',type=int,default=5)
parser.add_argument('--n-timesteps',type=int,default=100000)

args = parser.parse_args()

'''Observations on stable-baselines:

    1. discrete action space - categorical disrtribution
       continuous action space - DiagGaussian distribution

    2. there are 2 neural networks: -feature extractor MLp
                                    -output nn.Linear

        -feature extractor consists of 2 networks: policy and value with architecture: 
            Linear(28,64)->Tanh()->Linear(64,64)->Tanh()

        -output = nn.Linear(64,n_actions)

        I am not adding noise to parameters of output layer, that's why i should only modify extractor Mlp

'''


def train_test(seed, env_name, total_timesteps,param_noise):
    env = gym.make(env_name)

    env.seed(seed)
    th.manual_seed(seed)

    model = PPO('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=total_timesteps, param_noise=param_noise)

    # after every training stage there is testing on 5 episodes

    rewards_batch = []
    # making a post-evaluation
    for i in range(5):
        episode_rewards = 0
        obs = env.reset()
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            # env.render()
            if done:
                break

        rewards_batch.append(episode_rewards)

    return rewards_batch



noisy_avg = []
all_noisy = []

for i in range(args.n_seeds):

    noisy_reward = train_test(i,args.env,total_timesteps=args.n_timesteps,param_noise=True)

    all_noisy.append(noisy_reward)
    noisy_avg.append(np.mean(noisy_reward))


print("_________Results of noisy experiment_________")
print('__________All results:________')
print(all_noisy)
print('_________Average results_______')
print(noisy_avg)





