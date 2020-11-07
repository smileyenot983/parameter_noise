import gym
import pybulletgym
import numpy as np
import torch as th
import argparse

# from stable_baselines3 import PPO
# from stable_baselines3.ppo import ppo as PPO
from stable_baselines3.ppo.ppo import PPO

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


def train_test(seed, env_name, total_timesteps):
    env = gym.make(env_name)

    env.seed(seed)
    th.manual_seed(seed)

    model = PPO('MlpPolicy',env,verbose=0)
    model.learn(total_timesteps=total_timesteps)

    # testing

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


unnoisy_avg = []

all_unnoisy = []

for i in range(args.n_seeds):
    original_reward = train_test(i,args.env,total_timesteps=args.n_timesteps)


    all_unnoisy.append(original_reward)
    unnoisy_avg.append(np.mean(original_reward))

print("_________Results of unnoisy experiment_________")
print('__________All results:________')
print(all_unnoisy)

print('_________Average results_______')
print(unnoisy_avg)


