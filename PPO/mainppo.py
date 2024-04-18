#This code is built from/modified from Eric Yang Yu's tutorial: Coding PPO from Scratch with PyTorch
#https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8 (https://github.com/ericyangyu/PPO-for-Beginners)
import gymnasium as gym
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from ppo import PPO
from network import FeedForwardNN, FeedForwardActNN
from eval_policy import eval_policy

def train(env, hyperparameters, actor_model, critic_model):
    print(f"Training", flush=True)

    # Create a model for PPO.
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    model.learn(total_timesteps=5e6)

    episodearr = np.array(model.logger['statetime'])
    avgreturn = np.array(model.logger['statreturn'])
    avglength = np.array(model.logger['statlength'])

    # print(episodearr)
    # print(avgreturn)

    stackedarray = np.vstack((episodearr, avgreturn, avglength))
    stackedarray = stackedarray.T

    np.savetxt('./PPOResults.csv', stackedarray, delimiter=',')

    plt.figure()
    plt.plot(episodearr, avgreturn)
    plt.title('Average Return vs Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.show(block=False)

    plt.figure()
    plt.plot(episodearr, avglength)
    plt.title('Average Length of Episode vs Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Average Length of Episode')
    plt.show()



def test(env, actor_model):
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = FeedForwardActNN(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    eval_policy(policy=policy, env=env, render=True)

def main():
    hyperparameters = {
                'timesteps_per_batch': 2048, 
                'max_timesteps_per_episode': 200, 
                'gamma': 0.99, 
                'n_updates_per_iteration': 10,
                'lr': 3e-4, 
                'clip': 0.2,
                'render': True,
                'render_every_i': 10
            }

    trainingmode = False

    if trainingmode:
        env = gym.make('InvertedDoublePendulum-v4')
        train(env=env, hyperparameters=hyperparameters, actor_model='', critic_model='')
    else:
        env = gym.make('InvertedDoublePendulum-v4', render_mode='human')
        test(env=env, actor_model='./ppo_actor.pth')

if __name__ == '__main__':
    main()