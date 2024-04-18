#This code was built off/modified from the template from: https://github.com/johnnycode8/gym_solutions/blob/main/cartpole_q.py
from __future__ import annotations

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import pickle
import random
import pandas as pd


def run(is_training=True, render=False):
    env = gym.make('InvertedDoublePendulum-v4')  # , render_mode='human')
    # actions can be either -1 or 1 (-1 to go left and 1 to go right)
    # Divide position, velocity, pole angle, and pole angular velocity into segments
    pos_space = np.linspace(-1.2, 1.2, 10)  # state 0: cart position
    sine_theta1 = np.linspace(-1, 1, 10)  # state 1 the sine of the angle between cart and first pole
    sine_theta2 = np.linspace(-1, 1, 10)  # state 2: sine of angle between two poles
    cosine_theta1 = np.linspace(-1, 1, 10)  # state 3: cosine of the angle between cart and first pole
    cosine_theta2 = np.linspace(-1, 1, 10)  # state 4: cosine angle between the two poles
    vel_space = np.linspace(-8, 8, 10)  # state 5: cart velocity
    ang_vel_omega1 = np.linspace(-10, 10, 10)  # state 6: angular velo between the cart and first pole
    ang_vel_omega2 = np.linspace(-10, 10, 10)  # state 7: angular velo between poles
    constraint1_space = np.linspace(-10, 10, 10)  # state 8
    constraint2_space = np.linspace(-10, 10, 10)  # state 9
    constraint3_space = np.linspace(-10, 10, 10)  # state 10
    action_space = np.linspace(-1, 1, 2)
    if (is_training):
        q = np.zeros((len(pos_space) + 1, len(sine_theta1) + 1, len(sine_theta2) + 1, len(cosine_theta1) + 1,
                      len(cosine_theta2) + 1, len(vel_space) + 1, len(ang_vel_omega1) + 1, len(ang_vel_omega2) + 1,
                      len(constraint1_space) + 1, len(constraint2_space) + 1, len(constraint3_space) + 1,
                      len(action_space)))  # init a
        # 11x11x11x11x2 array
    else:
        f = open('InvertedDoublePendulum.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.1  # alpha or learning rate
    discount_factor_g = 0.99  # gamma or discount factor.

    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.00001  # epsilon decay rate
    epsilon_decay_rate = 0.00001
    rng = np.random.default_rng()  # random number generator
    alive_per_episode = []
    rewards_per_episode = []
    save_eps = []
    episodes = 160000
    i = 0

    for i in range(episodes):
        # while True:

        state = env.reset(seed=7)[0]
        state_p = np.digitize(state[0], pos_space)
        state_sine1 = np.digitize(state[1], sine_theta1)
        state_sine2 = np.digitize(state[2], sine_theta2)
        state_cosine1 = np.digitize(state[3], cosine_theta1)
        state_cosine2 = np.digitize(state[4], cosine_theta2)
        state_vel = np.digitize(state[5], vel_space)
        state_omega1 = np.digitize(state[6], ang_vel_omega1)
        state_omega2 = np.digitize(state[7], ang_vel_omega2)
        state_con1 = np.digitize(state[8], constraint1_space)
        state_con2 = np.digitize(state[9], constraint2_space)
        state_con3 = np.digitize(state[10], constraint3_space)

        done = False  # True when reached goal

        rewards = 0
        alive = 0
        while (not done and rewards < 10000):

            if is_training and rng.random() < epsilon:
                # Choose random action  (0=go left, 1=go right)
                action = env.action_space.sample()
                # print(action)
            else:
                action = np.argmax(q[state_p, state_sine1, state_sine2, state_cosine1, state_cosine2, state_vel,
                                   state_omega1, state_omega2, state_con1, state_con2, state_con3:])

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated  # or truncated
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_sine1 = np.digitize(new_state[1], sine_theta1)
            new_state_sine2 = np.digitize(new_state[2], sine_theta2)
            new_state_cosine1 = np.digitize(new_state[3], cosine_theta1)
            new_state_cosine2 = np.digitize(new_state[4], cosine_theta2)
            new_state_vel = np.digitize(new_state[5], vel_space)
            new_state_omega1 = np.digitize(new_state[6], ang_vel_omega1)
            new_state_omega2 = np.digitize(new_state[7], ang_vel_omega2)
            new_state_con1 = np.digitize(new_state[8], constraint1_space)
            new_state_con2 = np.digitize(new_state[9], constraint2_space)
            new_state_con3 = np.digitize(new_state[10], constraint3_space)
            new_action = np.digitize(action, action_space) - 1

            if is_training:
                q[state_p, state_sine1, state_sine2, state_cosine1, state_cosine2, state_vel, state_omega1,
                  state_omega2, state_con1, state_con2, state_con3, new_action] = \
                    q[
                        state_p, state_sine1, state_sine2, state_cosine1, state_cosine2, state_vel, state_omega1,
                        state_omega2, state_con1, state_con2, state_con3, new_action] + learning_rate_a * (
                            reward + discount_factor_g * np.max(
                        q[new_state_p, new_state_sine1, new_state_sine2, new_state_cosine1, new_state_cosine2,
                        new_state_vel, new_state_omega1, new_state_omega2, new_state_con1, new_state_con2,
                        new_state_con3, :]) - q[state_p, state_sine1, state_sine2, state_cosine1, state_cosine2,
                                                state_vel, state_omega1, state_omega2, state_con1, state_con2,
                                                state_con3, new_action])

            state = new_state
            state_p = new_state_p
            state_sine1 = new_state_sine1
            state_sine2 = new_state_sine2
            state_cosine1 = new_state_cosine1
            state_cosine2 = new_state_cosine2
            state_vel = new_state_vel
            state_omega1 = new_state_omega1
            state_omega2 = new_state_omega2
            state_con1 = new_state_con1
            state_con2 = new_state_con2
            state_con3 = new_state_con3

            rewards += reward
            alive = alive + 1
            if not is_training and rewards % 100 == 0:
                print(f'Episode: {i}  Rewards: {rewards}')

        alive_per_episode.append(alive)
        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode) - 100:])
        mean_alive = np.mean(alive_per_episode[len(alive_per_episode) - 100:])
        if is_training and i % 100 == 0:
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f} Mean Alive {mean_alive:0.1f}')

        # if mean_rewards > 1000:
        #    break
        epsilon = max(epsilon - epsilon_decay_rate, 0.1)
        save_eps.append(epsilon)
        i += 1

    env.close()

    # Save Q table to file
    # if is_training:
    #  f = open('InvertedDoublePendulum.pkl', 'wb')
    #  pickle.dump(q, f)
    # f.close()

    mean_rewards = []
    mean_alive = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t - 100):(t + 1)]))
        mean_alive.append(np.mean(alive_per_episode[max(0, t - 100):(t + 1)]))
    df = pd.DataFrame(data={"reward": mean_rewards, "Eps": save_eps,"Alive": mean_alive
    })
    df.to_csv("q_learn_score_file7.csv", sep=',', index=True)
    # plt.savefig(f'InvertedDoublePendulum.png')


if __name__ == '__main__':
    run(is_training=True, render=False)

# run(is_training=False, render=True)
