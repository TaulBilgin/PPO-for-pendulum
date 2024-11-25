import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import copy
import numpy as np
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class GaussianActor_mu(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, log_std=0):
		super(GaussianActor_mu, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.mu_head.weight.data.mul_(0.1)
		self.mu_head.bias.data.mul_(0.0)

		self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

	def forward(self, state):
		a = torch.relu(self.l1(state))
		a = torch.relu(self.l2(a))
		mu = (torch.tanh(self.mu_head(a)) * 2)
		return mu

	def get_dist(self,state):
		mu = self.forward(state)
		action_log_std = self.action_log_std.expand_as(mu)
		action_std = torch.exp(action_log_std)

		dist = Normal(mu, action_std)
		return dist

	def deterministic_act(self, state):
		return self.forward(state)

class Critic(nn.Module):
	def __init__(self, state_dim,net_width):
		super(Critic, self).__init__()

		self.C1 = nn.Linear(state_dim, net_width)
		self.C2 = nn.Linear(net_width, net_width)
		self.C3 = nn.Linear(net_width, 1)

	def forward(self, state):
		v = torch.tanh(self.C1(state))
		v = torch.tanh(self.C2(v))
		v = self.C3(v)
		return v

def evaluate_policy(env, agent, turns):
	total_scores = 0
	for j in range(turns):
		s, info = env.reset()
		done = False
		while not done:
			a, logprob_a = agent.select_action(s, deterministic=True) # Take deterministic actions when evaluation
			s_next, r, dw, tr, info = env.step(a)
			done = (dw or tr)

			total_scores += r
			s = s_next

	return total_scores/turns

class PPO_agent():
	def __init__(self, state_dim=3, action_dim=1, net_width=150, T_horizon=2048,
                 a_lr=2e-4, c_lr=2e-4, gamma=0.99, lambd=0.95, clip_rate=0.2,
                 K_epochs=10, a_optim_batch_size=64, c_optim_batch_size=64,
                 entropy_coef=1e-3, entropy_coef_decay=0.99, l2_reg=1e-3):
	
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.net_width = net_width
		self.T_horizon = T_horizon
		
		# Training parameters
		self.gamma = gamma
		self.lambd = lambd
		self.clip_rate = clip_rate
		self.K_epochs = K_epochs
		self.a_optim_batch_size = a_optim_batch_size
		self.c_optim_batch_size = c_optim_batch_size
		self.entropy_coef = entropy_coef
		self.entropy_coef_decay = entropy_coef_decay
		self.l2_reg = l2_reg

		# Choose distribution for the actor
		self.actor = GaussianActor_mu(self.state_dim, self.action_dim, self.net_width).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

		# Build Critic
		self.critic = Critic(self.state_dim, self.net_width).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

		# Build Trajectory holder
		self.s_hoder = torch.zeros((self.T_horizon, self.state_dim)).to(device)
		self.a_hoder = torch.zeros((self.T_horizon, self.action_dim)).to(device)
		self.r_hoder = torch.zeros((self.T_horizon, 1)).to(device)
		self.s_next_hoder = torch.zeros((self.T_horizon, self.state_dim)).to(device)
		self.logprob_a_hoder = torch.zeros((self.T_horizon, self.action_dim)).to(device)
		self.done_hoder = torch.zeros((self.T_horizon, 1)).to(device)
		self.dw_hoder = torch.zeros((self.T_horizon, 1)).to(device)

	def put_data(self, now_state, action, reward, next_state, logprob_a, done, dw, idx):
		self.s_hoder[idx] = torch.from_numpy(now_state).to(device)
		self.a_hoder[idx] = torch.from_numpy(action).to(device)
		self.s_next_hoder[idx] = torch.from_numpy(next_state).to(device)
		self.logprob_a_hoder[idx] = torch.from_numpy(logprob_a).to(device)
		
		# Handle scalar values by converting to tensors directly
		self.r_hoder[idx] = torch.tensor([reward], dtype=torch.float32).to(device)
		self.done_hoder[idx] = torch.tensor([done], dtype=torch.float32).to(device)
		self.dw_hoder[idx] = torch.tensor([dw], dtype=torch.float32).to(device)

	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			if deterministic:
				# only used when evaluate the policy
				a = self.actor.deterministic_act(state)
				return a.cpu().numpy()[0], None  # action is in shape (adim, 0)
			else:
				# only used when interact with the env
				dist = self.actor.get_dist(state)
				a = dist.sample()
				a = torch.clamp(a, -2, 2)
				logprob_a = dist.log_prob(a).cpu().numpy().flatten()
				return a.cpu().numpy()[0], logprob_a # both are in shape (adim, 0)
	def train(self):
		self.entropy_coef*=self.entropy_coef_decay

		'''Prepare PyTorch data from Numpy data'''
		now_state = self.s_hoder
		action = self.a_hoder
		reward = self.r_hoder
		next_state = self.s_next_hoder
		logprob_action = self.logprob_a_hoder
		done = self.done_hoder
		dw = self.dw_hoder

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			value = self.critic(now_state)
			next_value = self.critic(next_state)

			'''dw for TD_target and Adv'''
			deltas = reward + self.gamma * next_value * (1-dw) - value
			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			'''done for GAE'''
			for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (1-mask)
				adv.append(advantage)
			adv.reverse()
			adv = copy.deepcopy(adv[0:-1])
			adv = torch.tensor(adv).unsqueeze(1).float().to(device)
			td_target = adv + value
			adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps


		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(now_state.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(now_state.shape[0] / self.c_optim_batch_size))
		for i in range(self.K_epochs):
			#Shuffle the trajectory, Good for training
			perm = np.arange(now_state.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(device)
			clone_now_state, clone_action, clone_td_target, clone_adv, clone_logprob_action = \
				now_state[perm].clone(), action[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_action	[perm].clone()

			'''update the actor'''
			for i in range(a_optim_iter_num):
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, clone_now_state.shape[0]))
				distribution = self.actor.get_dist(clone_now_state[index])
				dist_entropy = distribution.entropy().sum(1, keepdim=True)
				
				# Get log probabilities of the actions under current policy
				logprob_a_now = distribution.log_prob(action[index])
				
				# Calculate probability ratio between new and old policies
				# ratio = π_new(a|s) / π_old(a|s)
				# Using log probabilities: ratio = exp(log(π_new) - log(π_old))
				ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - 
								 logprob_action[index].sum(1,keepdim=True))

				# Calculate surrogate objectives for PPO's clipped objective
				surr1 = ratio * adv[index]  # Standard policy gradient
				# Clip the ratio to prevent too large policy updates
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				
				# Calculate final loss (negative because we want to maximize the objective)
				# Take minimum of standard and clipped objectives (PPO's pessimistic bound)
				# Subtract entropy term to encourage exploration
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				# Perform gradient update
				self.actor_optimizer.zero_grad()  # Clear previous gradients
				a_loss.mean().backward()  # Compute gradients
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)  # Clip gradients to prevent explosion
				self.actor_optimizer.step()  # Update network parameters

			'''update the critic'''
			for i in range(c_optim_iter_num):
				index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, clone_now_state.shape[0]))
				c_loss = (self.critic(clone_now_state[index]) - td_target[index]).pow(2).mean()
				for name,param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()

def main():
	env = gym.make('Pendulum-v1')
	eval_env = gym.make('Pendulum-v1')
	agent = PPO_agent()
	env_seed = 0
	best_score = -200
    
	traj_lenth, total_steps, total_score = 0, 0, 0
	while total_steps < 5e7:
		s, info = env.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed
		
		done = False

		'''Interact & trian'''
		while not done:
			'''Interact with Env'''
			a, logprob_a = agent.select_action(s, deterministic=False) # use stochastic when training
			# act = Action_adapter(a, 2) #[0,1] to [-max,max]
			s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
			
			done = (dw or tr)
			total_score += r

			'''Store the current transition'''
			agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx = traj_lenth)
			s = s_next

			traj_lenth += 1
			total_steps += 1

			'''Update if its time'''
			if traj_lenth % 2048 == 0:
				agent.train()
				traj_lenth = 0

			'''Record & log'''
			if total_steps % 5e3 == 0: # evaluate the policy for 1 times, and get averaged result
				score = evaluate_policy(eval_env, agent, turns=1)
				print('EnvName:','Pendulum-v1','seed:',"0",'steps: {}k'.format(int(total_steps/1000)),'score:', score)
				if int(score) > best_score:
					best_score = int(score)
					torch.save(agent.actor.state_dict(), f"Pendulum{best_score}.pt")

				total_score = 0

		env.close()
	return
	
main()


