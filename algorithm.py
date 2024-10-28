import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
import logging
import time

# Configure logging to write to a file named `training_log.log`
logging.basicConfig(
    filename='training.log',  # Specify the log file name
    filemode='w',  # Use 'w' to overwrite the file each run, 'a' to append
    level=logging.DEBUG,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)

class PowerFlowEnv(gym.Env):
    def __init__(self, bus_data, line_data, max_steps=2000):
        super(PowerFlowEnv, self).__init__()
        self.bus_data = bus_data  # List of bus data (P_demand, Q_demand, voltage limits)
        self.line_data = line_data  # List of line data (from_bus, to_bus, resistance, reactance)
        self.max_steps = max_steps  # Maximum number of steps before the episode ends
        self.current_step = 0  # Step counter to track the number of steps in the current episode
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(bus_data),), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(len(bus_data),), dtype=np.float32)

    def reset(self):
        # Reset the environment's state
        self.current_step = 0  # Reset step counter for a new episode
        self.state = np.random.uniform(low=0.95, high=1.05, size=len(self.bus_data))  # Initial voltages
        return self.state

    def calculate_power_flow(self):
        # Calculate the total power loss in the system
        power_loss = 0.0
        for from_bus, to_bus, r, x in self.line_data:
            V_from = self.state[from_bus]
            V_to = self.state[to_bus]
            delta_v = V_from - V_to  # Voltage difference between buses
            impedance = complex(r, x)
            current = delta_v / impedance  # Calculate current based on Ohm's law
            power_loss += abs(V_from * np.conj(current))  # Accumulate power loss
        return power_loss

    def step(self, action):
        # Apply action to update voltages
        self.state += action * 0.1
        power_loss = self.calculate_power_flow()
        reward = -power_loss  # Reward is the negative of power loss to encourage minimizing losses

        # Increment the step counter
        self.current_step += 1

        # Define termination conditions
        done = False
        if self.current_step >= self.max_steps:
            done = True  # End the episode if the maximum steps are reached
        elif power_loss > 10.0:  # Example threshold, adjust based on your system's specifics
            done = True  # End if power loss is too high
        elif np.any((self.state < 0.9) | (self.state > 1.1)):  # Voltage limits
            done = True  # End if any bus voltage goes out of acceptable bounds

        return self.state, reward, done, {}

    def calculate_reward(self):
        # Calculate reward separately if needed
        return -self.calculate_power_flow()


# Actor Network: This network generates actions based on the observed state
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        
        # Neural network layers for the actor:
        # Fully connected layers process input data to output actions
        # Layers designed for increasing abstraction with a progressively reduced size

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # Input to first layer with 128 nodes
            nn.ReLU(),  # ReLU activation introduces non-linearity
            nn.Linear(128, 64),  # Second layer with 64 nodes, reducing size for more specific features
            nn.ReLU(),  # Another non-linearity
            nn.Linear(64, output_dim),  # Output layer to match action dimension
            nn.Tanh()  # Tanh bounds actions between -1 and 1
        )

    def forward(self, state):
        return self.fc(state)  # Forward pass outputs action based on current state


# Critic Network: This network evaluates how good a given state is
class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        
        # Neural network layers for the critic:
        # Evaluates the value of the current state for policy improvement

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # First layer matches input size to 128 nodes
            nn.ReLU(),  # ReLU activation for non-linear processing
            nn.Linear(128, 64),  # Second layer with 64 nodes for deeper feature extraction
            nn.ReLU(),
            nn.Linear(64, 1)  # Final layer outputs a single value (state value estimation)
        )

    def forward(self, state):
        return self.fc(state)  # Outputs a scalar value representing the state's value


# PPO Agent: Implements Proximal Policy Optimization with Clipped Objective
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, eps_clip=0.2):
        
        # Initialize actor and critic networks
        self.actor = ActorNetwork(state_dim, action_dim)  # Actor network to generate actions
        self.critic = CriticNetwork(state_dim)  # Critic network to evaluate state value
        
        # Optimizer for updating network weights; combines actor and critic params
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma  # Discount factor for rewards (future rewards weighting)
        self.eps_clip = eps_clip  # Clipping parameter to limit policy updates

    def select_action(self, state):
        # Generates action using actor network and converts to numpy
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
        action = self.actor(state)  # Actor generates action from the state
        return action.detach().numpy().flatten()  # Detach from graph and flatten to use as action in env

    def compute_advantages(self, rewards, values, next_value, dones):
        # Calculate advantages for PPO policy update
        # The advantage tells how much better an action was than the average action in that state

        advantages = []
        advantage = 0
        for i in reversed(range(len(rewards))):
            # Temporal Difference (TD) error, accounting for gamma and future reward
            td_error = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            advantage = td_error + self.gamma * advantage * (1 - dones[i])  # Accumulate advantages over time
            advantages.insert(0, advantage)  # Insert at start to keep reverse order
            next_value = values[i]
        return advantages  # Return list of advantages

    def update(self, states, actions, rewards, dones, values):
        # Convert lists to tensors with consistent dimensions
        states = torch.FloatTensor(np.array(states))  # Convert list of arrays to a single numpy array first
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)  # Ensure rewards have shape [N, 1]
        dones = torch.FloatTensor(dones)
        values = torch.FloatTensor(values).unsqueeze(-1)  # Ensure values have shape [N, 1]

        next_value = self.critic(states[-1].unsqueeze(0))  # Shape adjustment for the last state
        advantages = self.compute_advantages(rewards, values, next_value, dones)  # Compute advantages

        for _ in range(10):  # Multiple epochs to refine networks
            # Actor updates: calculate action probabilities
            current_actions = self.actor(states)
            action_log_probs = -((current_actions - actions) ** 2).mean()  # Negative log-probabilities (for simplicity)
            ratio = torch.exp(action_log_probs - action_log_probs.detach())  # New-to-old policy ratio

            # PPO Clipping for actor loss
            advantages_tensor = torch.FloatTensor(advantages).unsqueeze(-1)  # Match dimensions for advantages
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()  # Minimize clipped objective

            # Critic loss using MSE between predicted and actual rewards
            critic_loss = nn.MSELoss()(self.critic(states).squeeze(), rewards.squeeze())

            # Combined actor-critic loss and optimizer step
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# Main Training Loop: runs through episodes and updates agent after each episode
def train(agent, env, episodes=500):
    for episode in range(episodes):

        logging.info("_" * 120)  # Adds a line with 120 underscores
        logging.info(f"Starting Episode {episode + 1}")
        start_time = time.time()

        state = env.reset()  # Reset environment to start of episode
        done = False
        episode_reward = 0  # Track cumulative reward for the episode
        states, actions, rewards, dones, values = [], [], [], [], []  # Initialize buffers

        # Run the episode
        while not done:
            logging.debug("Selecting action...")
            action = agent.select_action(state)  # Agent selects an action based on state
            logging.debug(f"Action selected: {action}")

            logging.debug("Taking step in environment...")
            next_state, reward, done, _ = env.step(action)  # Environment returns next state and reward
            logging.debug(f"Step taken, reward: {reward}, done: {done}")

            logging.debug("Getting value from critic...")
            value = agent.critic(torch.FloatTensor(state))  # Get state value from critic
            logging.debug(f"Value obtained: {value.item()}")

            # Store episode data for policy update
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            # Update state and cumulative reward

            state = next_state
            episode_reward += reward

        # After episode ends, update the PPO agent
        logging.info(f"Episode {episode + 1} finished, starting agent update...")
        agent.update(states, actions, rewards, dones, values)
        logging.info(f"Episode {episode + 1}, Reward: {episode_reward}, Time: {time.time() - start_time:.2f} seconds")

        print(f"Episode {episode + 1}, Reward: {episode_reward}")  # Track reward to monitor learning progress


# Load the MATLAB data file
data = loadmat('powerflow_data.mat')
bus_data = data['bus_data']  # Extract bus data from MATLAB file
branch_data = data['branch_data']  # Extract branch data (line data)

# Format data to match PowerFlowEnv requirements
bus_data_list = [(int(row[0]), row[1], row[2]) for row in bus_data]  # Adjusted for bus data format (BusNum, LoadMW, LoadMVAR)
line_data_list = [(int(row[0])-1, int(row[1])-1, row[2], row[3]) for row in branch_data]  # Adjusted for line data format (from_bus, to_bus, resistance, reactance)

# Instantiate the environment using the MATLAB-loaded data
env = PowerFlowEnv(bus_data=bus_data_list, line_data=line_data_list)
state_dim = env.observation_space.shape[0]  # State dimension based on the environment's observation space
action_dim = env.action_space.shape[0]  # Action dimension based on the environment's action space

# Initialize the PPO Agent
agent = PPOAgent(state_dim, action_dim)

# Train the agent using the environment with the MATLAB data
train(agent, env, episodes=50)
