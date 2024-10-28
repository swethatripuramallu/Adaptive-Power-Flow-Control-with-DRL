import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim

# Define PowerFlowEnv environment
class PowerFlowEnv(gym.Env):
    def __init__(self, bus_data, line_data):
        super(PowerFlowEnv, self).__init__()
        self.bus_data = bus_data  # List of bus data (P_demand, Q_demand, voltage limits)
        self.line_data = line_data  # List of line data (from_bus, to_bus, resistance, reactance)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(bus_data),), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(len(bus_data),), dtype=np.float32)

    def reset(self):
        # Initialize voltages at each bus as part of the state
        self.state = np.random.uniform(low=0.95, high=1.05, size=len(self.bus_data))  
        return self.state

    def calculate_power_flow(self):
        power_loss = 0.0
        for from_bus, to_bus, r, x in self.line_data:
            V_from = self.state[from_bus]
            V_to = self.state[to_bus]
            delta_theta = np.angle(V_from) - np.angle(V_to)  # Angle difference between buses
            current = (V_from - V_to) / complex(r, x)  # Current calculation using Ohm's law
            power_loss += abs(V_from * np.conj(current))  # Power loss calculation
        return power_loss

    def step(self, action):
        self.state += action * 0.1  # Adjust voltages based on actions
        power_loss = self.calculate_power_flow()
        reward = -power_loss  # Reward is the negative of power loss
        done = False  # This problem does not have a terminal state
        return self.state, reward, done, {}

    def calculate_reward(self):
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
        # Update actor and critic using PPO clipping
        # States, actions, rewards, dones, values are stored from the episode

        states = torch.FloatTensor(states)  # Convert states to tensor
        actions = torch.FloatTensor(actions)  # Convert actions to tensor
        rewards = torch.FloatTensor(rewards)  # Convert rewards to tensor
        dones = torch.FloatTensor(dones)  # Convert dones to tensor (binary done signals)
        values = torch.FloatTensor(values)  # Convert values (state-value estimates) to tensor

        next_value = self.critic(states[-1])  # Estimate value of the last state
        advantages = self.compute_advantages(rewards, values, next_value, dones)  # Calculate advantages

        # Perform multiple epochs to refine actor and critic networks
        for _ in range(10):
            # Actor updates: determine action probabilities
            current_actions = self.actor(states)
            action_log_probs = -((current_actions - actions)**2).mean()  # Negative log-probabilities (for simplicity)
            ratio = torch.exp(action_log_probs - action_log_probs.detach())  # Ratio of new to old policies

            # Calculate actor loss with PPO Clipping (limits policy updates for stability)
            surr1 = ratio * torch.FloatTensor(advantages)
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * torch.FloatTensor(advantages)
            actor_loss = -torch.min(surr1, surr2).mean()  # Minimize clipped objective

            # Critic loss: minimizes MSE between predicted and actual rewards
            critic_loss = nn.MSELoss()(self.critic(states).squeeze(), rewards)

            # Backpropagation for combined actor-critic loss
            loss = actor_loss + 0.5 * critic_loss  # Weighted sum of losses
            self.optimizer.zero_grad()  # Zero gradients for optimizer
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Optimizer update step


# Main Training Loop: runs through episodes and updates agent after each episode
def train(agent, env, episodes=500):
    for episode in range(episodes):
        state = env.reset()  # Reset environment to start of episode
        done = False
        episode_reward = 0  # Track cumulative reward for the episode
        states, actions, rewards, dones, values = [], [], [], [], []  # Initialize buffers

        # Run the episode
        while not done:
            action = agent.select_action(state)  # Agent selects an action based on state
            next_state, reward, done, _ = env.step(action)  # Environment returns next state and reward
            value = agent.critic(torch.FloatTensor(state))  # Get state value from critic

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
        agent.update(states, actions, rewards, dones, values)
        print(f"Episode {episode + 1}, Reward: {episode_reward}")  # Track reward to monitor learning progress

# # Instantiate environment and agent
# env = PowerFlowEnv(bus_data=[(i, 1.0, 0.5) for i in range(10)], line_data=[(i, i+1, 0.1, 0.1) for i in range(9)])
# state_dim = env.observation_space.shape[0]  # State dimension
# action_dim = env.action_space.shape[0]  # Action dimension
# agent = PPOAgent(state_dim, action_dim)  # Initialize PPO agent

# # Train the agent with the environment
# train(agent, env, episodes=500)  # Start the training process