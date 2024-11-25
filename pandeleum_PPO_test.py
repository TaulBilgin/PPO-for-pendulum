import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, log_std=0):
		super(Actor, self).__init__()

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

	def deterministic_act(self, state):
		return self.forward(state)

def select_action(state, actor):
	with torch.no_grad():
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		# only used when evaluate the policy.Making the performance more stable
		a = actor.deterministic_act(state)
		return a.cpu().numpy()[0], None  # action is in shape (adim, 0)

env = gym.make('Pendulum-v1', render_mode="human")
state_dim = 3  # Dimension of the state space
action_dim = 1  # Dimension of the action space
net_width = 150  # Width of the network (number of units in hidden layers)

# Seed everything for reproducibility
env_seed = 0
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Initialize the Actor network and move it to the device (GPU or CPU)
actor = Actor(state_dim, action_dim, net_width).to(device)

# Load the pre-trained model weights for the Actor network
actor.load_state_dict(torch.load("your model name")) # like "Pendulum-123.pt"

# Switch the Actor network to evaluation mode (disables dropout, etc.)
actor.eval()

run = 0  # Initialize episode counter
while run < 10:  # Run for 10 episodes
	# Reset the environment and get the initial state
	now_state = env.reset(seed=env_seed)[0]
	done = False
	step = 0

    # Interact with the environment for 200 steps (or until termination)
	while not done:  
		action, _ = select_action(now_state, actor)  # Select action using the actor network
		next_state, reward, done, truncated, info = env.step(action)  # Take the selected action in the environment
		now_state = next_state  # Update the current state to the next state
		step += 1
		done = (truncated or done)
        
