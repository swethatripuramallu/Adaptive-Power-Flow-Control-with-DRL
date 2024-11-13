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
    filename='training.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PowerFlowEnv(gym.Env):
    def __init__(self, bus_data, line_data, max_steps=24):
        super(PowerFlowEnv, self).__init__()
        self.bus_data = bus_data
        self.line_data = line_data
        self.max_steps = max_steps
        self.current_step = 0
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(bus_data),), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(len(bus_data),), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.state = np.random.uniform(low=0.95, high=1.05, size=len(self.bus_data))
        return self.state

    def calculate_power_flow(self):
        power_loss = 0.0
        for from_bus, to_bus, r, x in self.line_data:
            V_from = self.state[from_bus]
            V_to = self.state[to_bus]
            delta_v = V_from - V_to
            impedance = complex(r, x)
            current = delta_v / impedance
            power_loss += abs(V_from * np.conj(current))
        return power_loss

    def step(self, action):
        self.state += action * 0.1
        power_loss = self.calculate_power_flow()
        reward = -power_loss

        self.current_step += 1
        done = False
        if self.current_step >= self.max_steps or power_loss > 10.0 or np.any((self.state < 0.9) | (self.state > 1.1)):
            done = True

        return self.state, reward, done, {}

    def calculate_reward(self):
        return -self.calculate_power_flow()

# Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.fc(state)

# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.fc(state)

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, eps_clip=0.2):
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state)
        return action.detach().cpu().numpy().flatten()

    def compute_advantages(self, rewards, values, next_value, dones):
        advantages = []
        advantage = 0
        for i in reversed(range(len(rewards))):
            td_error = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            advantage = td_error + self.gamma * advantage * (1 - dones[i])
            advantages.insert(0, advantage)
            next_value = values[i]
        return advantages

    def update(self, states, actions, rewards, dones, values):
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
        dones = torch.FloatTensor(dones).to(device)
        values = torch.FloatTensor(values).unsqueeze(-1).to(device)

        next_value = self.critic(states[-1].unsqueeze(0))
        advantages = self.compute_advantages(rewards.cpu(), values.cpu(), next_value.cpu(), dones.cpu())

        for _ in range(10):
            current_actions = self.actor(states)
            action_log_probs = -((current_actions - actions) ** 2).mean()
            ratio = torch.exp(action_log_probs - action_log_probs.detach())
            advantages_tensor = torch.FloatTensor(advantages).unsqueeze(-1).to(device)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(self.critic(states).squeeze(), rewards.squeeze())
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Training Loop
def train(agent, env, episodes=50):
   # Open the log file in append mode
    with open("power_flow_data.log", "a") as log_file:
        if log_file.tell() == 0:  # Check if the file is empty to write a header
            log_file.write("Episode, Step, Bus_Powers, Reward\n")  # Write the header only if the file is empty

        for episode in range(episodes):
            logging.info("_" * 120)
            logging.info(f"Starting Episode {episode + 1}")
            start_time = time.time()

            state = env.reset()
            done = False
            episode_reward = 0
            states, actions, rewards, dones, values = [], [], [], [], []
            step = 0

            while not done:
                logging.debug("Selecting action...")
                action = agent.select_action(state)
                logging.debug(f"Action selected: {action}")

                logging.debug("Taking step in environment...")
                next_state, reward, done, _ = env.step(action)
                logging.debug(f"Step taken, reward: {reward}, done: {done}")

                logging.debug("Getting value from critic...")
                value = agent.critic(torch.FloatTensor(state).to(device))
                logging.debug(f"Value obtained: {value.item()}")

                bus_powers = state.tolist()
                log_file.write(f"{episode + 1}, {step + 1}, {bus_powers}, {reward}\n")

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())

                state = next_state
                episode_reward += reward
                step += 1

            logging.info(f"Episode {episode + 1} finished, starting agent update...")
            agent.update(states, actions, rewards, dones, values)
            logging.info(f"Episode {episode + 1}, Reward: {episode_reward}, Time: {time.time() - start_time:.2f} seconds")
            print(f"Episode {episode + 1}, Reward: {episode_reward}")

# Load the MATLAB data file
data = loadmat('powerflow_data.mat')
bus_data = data['bus_data']
branch_data = data['branch_data']
bus_data_list = [(int(row[0]), row[1], row[2]) for row in bus_data]
line_data_list = [(int(row[0])-1, int(row[1])-1, row[2], row[3]) for row in branch_data]

env = PowerFlowEnv(bus_data=bus_data_list, line_data=line_data_list)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Initialize the PPO Agent
agent = PPOAgent(state_dim, action_dim)

# Train the agent using the environment with the MATLAB data
train(agent, env, episodes=50)
