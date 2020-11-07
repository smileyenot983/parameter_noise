import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str)
parser.add_argument('--n-seeds',type=int,default=5)
parser.add_argument('--n-timesteps',type=int,default=100000)

args = parser.parse_args()
# first running with noise
# print("_________Results of noisy experiment_________")
os.system("python stable-baselines3_noise/testing_PPO.py --env "+args.env+" --n-seeds " + str(args.n_seeds) + " --n-timesteps " + str(args.n_timesteps))
print("\n\n")
# first running with noise
# print("_________Results of unnoisy experiment_________")
os.system("python stable-baselines3_original/testing_PPO_original.py --env "+args.env+" --n-seeds " + str(args.n_seeds) + " --n-timesteps " + str(args.n_timesteps))