#This code is built from/modified from Eric Yang Yu's tutorial: Coding PPO from Scratch with PyTorch
#https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8 (https://github.com/ericyangyu/PPO-for-Beginners)
from network import FeedForwardNN, FeedForwardActNN
from torch.distributions import MultivariateNormal
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import time
import gymnasium as gym

class PPO:
    def __init__(self, env, **hyperparameters):
        #init hyperparams
        self._init_hyperparams(hyperparameters)

        #Get info about environment.
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        #Init the actor and critic networks.
        self.actor = FeedForwardActNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        

        self.cov_var = torch.full(size=(self.act_dim,), fill_value = 0.5)
        #cov matrix
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), lr = self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.lr)

        # self.actor_optim = Adam(self.actor.parameters(), lr = 1e-5)
        # self.critic_optim = Adam(self.critic.parameters(), lr = 5e-3)

        # This is for printing during runtime
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            'lr': 0,
            'numepisodes': [],
            'statreturn' : [],
            'statlength' : [],
            'statetime' : [],
        }

    def learn(self, total_timesteps):
        t_current = 0 #Current timestep
        i_current = 0

        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        while t_current < total_timesteps:
            
            #Do Rollout
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout(t_current)

            #GAE
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()

            t_current += np.sum(batch_lens)
            # Increment the number of iterations
            i_current += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_current
            self.logger['i_so_far'] = i_current

            #Normalize the advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            #minibatches
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []

            for _ in range(self.n_updates_per_iteration):
                # Learning Rate Annealing
                frac = (t_current - 1.0) / total_timesteps
                new_lr = self.lr * (1.0-frac)
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                # End of Learning Rate Annealing
                self.logger['lr'] = new_lr

                np.random.shuffle(inds)
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)
                    logratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()

                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    #Get Losses
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V.squeeze(), mini_rtgs.squeeze())
                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss - self.ent_coef * entropy_loss
                    

                    #calculate gradients and perform back propogation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm) # gradient clipping
                    self.actor_optim.step()

                    #gradients and back propogation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm) # gradient clipping
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())

                if approx_kl > self.target_kl:
                    break

            # Log actor loss
            avg_loss = sum(loss) / len(loss)
            self.logger['actor_losses'].append(avg_loss)

            #Print stats
            self._log_summary()
            if i_current % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')
        
    def _init_hyperparams(self, hyperparameters):
        self.timesteps_per_batch = 2048                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1000           # Max number of timesteps per episode
        self.n_updates_per_iteration = 10                # Number of times to update actor/critic per iteration
        self.lr = 3e-4                                 # Learning rate of actor optimizer
        self.gamma = 0.99                              # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.lam = 0.95                                 # Lambda Parameter for GAE 
        self.num_minibatches = 64                        # Number of mini-batches for Mini-batch Update
        self.ent_coef = 0.01                               # Entropy coefficient for Entropy Regularization
        self.target_kl = 0.02                           # KL Divergence threshold
        self.max_grad_norm = 0.5                        # Gradient Clipping threshold


        # Miscellaneous parameters
        self.render = False                             # If we should render during rollout
        self.save_freq = 10                             # How often we save in number of iterations
        self.deterministic = False                      # If we're testing, don't sample actions
        self.seed = 7								# Sets the seed of our program, used for reproducibility of results
        
        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def rollout(self, t_so_far):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        #episodic data
        ep_rews = []
        ep_vals = []
        ep_dones = []
        numepi = 0


        t = 0 # timesteps run so far this batch
        while t < self.timesteps_per_batch:
            #rewards for this episode
            ep_rews = []
            ep_vals = []
            ep_dones = []

            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                if self.render:
                    self.env.render()
                ep_dones.append(done)

                t += 1

                #collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                val = self.critic(obs) # Get Value from Critic Network

                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                #collect the reward, action and log probability
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    numepi += 1
                    break

            #collect episodic length & rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
        
        #reshape data as tensors
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten()

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        if not self.logger['numepisodes']:
            self.logger['numepisodes'].append(numepi)
        else:
            self.logger['numepisodes'].append(self.logger['numepisodes'][-1] + numepi)

        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones

    def get_action(self, obs):
        mean = self.actor(obs)

        #multivariate normal distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        #sample action from distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        #detach since action and log prob are tensors with computation graphs.
        return action.detach().numpy(), log_prob.detach()
    
    def evaluate(self, batch_obs, batch_acts):
        #Get value V for each obs
        V = self.critic(batch_obs).squeeze()

        #calc log probabilities of batch actions
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs, dist.entropy()
    
    def _log_summary(self):
        #Helper function keeping track of metrics
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        lr = self.logger['lr']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        #logging for curves
        self.logger['statreturn'].append(avg_ep_rews)
        self.logger['statlength'].append(avg_ep_lens)
        self.logger['statetime'].append(self.logger['numepisodes'][-1])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        numeps = str(self.logger['numepisodes'][-1])

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Episodes so far: {numeps}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

    #Using generalized advantage estimation to get advantage
    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)
