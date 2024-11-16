# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pypower.api import loadcase, runpf

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define PyPowerEnv class
class PyPowerEnv:
    def __init__(self, case_file, max_steps=24):
        """
        Initialize the PyPower environment.
        
        Parameters:
        - case_file: The MATPOWER case file (as a dictionary or filepath) to load.
        - max_steps: Maximum steps for an episode (24 steps for a 24-hour day).
        """
        self.case_file = case_file
        self.max_steps = max_steps
        self.current_step = 0
        self.mpc = self.load_case_file()
        self.state = self.get_initial_state()
        self.done = False

        # System parameters and battery SOC limits
        self.Q_pv_min, self.Q_pv_max = 0, 6.87  # MW for PV
        self.Q_w_min, self.Q_w_max = 0, 0.968  # MW for wind
        self.P_bss_min, self.P_bss_max = -2, 2  # MW for battery
        self.Q_bss_min, self.Q_bss_max = 0, 0.968  # MW for battery
        self.V_min, self.V_max = 0.94, 1.06  # Voltage limits in p.u.
        self.E_max, self.E_min = 3.6, 0.8  # MWh, energy capacity limits
        self.battery_energy = 2.0  # Initial battery SOC in MWh
        self.mu_b = 0.98  # Charging/discharging efficiency

    def load_case_file(self):
        """Load the MATPOWER case file."""
        if isinstance(self.case_file, dict):
            return self.case_file
        return loadcase(self.case_file)

    def get_initial_state(self):
        """Initialize the state with bus voltages."""
        self.mpc = self.load_case_file()
        results, success = runpf(self.mpc)
        if not success:
            raise RuntimeError("Initial power flow did not converge")
        return results['bus'][:, 7]  # Return initial bus voltages (magnitudes)

    def reset(self):
        """Reset the environment."""
        self.current_step = 0
        self.done = False
        self.battery_energy = 2.0
        self.state = self.get_initial_state()
        return self.state

    def update_soc(self, P_b):
        """Update the state of charge (SOC) of the battery."""
        self.battery_energy += self.mu_b * P_b
        self.battery_energy = min(max(self.battery_energy, self.E_min), self.E_max)

    def step(self, action):
        """Perform one step in the environment."""
        Q_pv = action[0:2]
        Q_w = action[2]
        P_bss = action[3]
        Q_bss = action[4]

        self.mpc['gen'][1, 2] = Q_pv[0]
        self.mpc['gen'][2, 2] = Q_pv[1]
        self.mpc['gen'][4, 2] = Q_w
        self.mpc['gen'][3, 1] = P_bss
        self.mpc['gen'][3, 2] = Q_bss

        self.update_soc(P_bss)

        results, success = runpf(self.mpc)
        if not success:
            self.done = True
            return self.state, -100.0, self.done, {"battery_energy": self.battery_energy}

        system_loss = np.sum(results['branch'][:, 14])
        reward = -system_loss

        self.state = results['bus'][:, 7]
        self.current_step += 1
        self.done = self.current_step >= self.max_steps

        return self.state, reward, self.done, {"battery_energy": self.battery_energy, "system_loss": system_loss}


# Actor and Critic Network definitions (same as before)
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, state):
        return self.fc(state)
    
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

# PPO Agent class (same as before)
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003,
                 gamma=0.99, eps_clip=0.1):
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) 
                                    + list(self.critic.parameters()), lr=lr)
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
        advantages = self.compute_advantages(rewards.cpu(), values.cpu(), 
                                             next_value.cpu(), dones.cpu())

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

def setup_environment_and_agent(case_file):
    """Set up the environment and PPO agent."""
    env = PyPowerEnv(case_file)
    state_dim = len(env.state)
    action_dim = 5
    agent = PPOAgent(state_dim, action_dim)
    return env, agent

def train_agent(agent, env, episodes=360):
    """Train the PPO agent on the environment."""
    step_log = []
    episode_log = []
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0

        for hour in range(24):  # Each episode has 24 steps
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            step_log.append(
                {
                    "Episode": episode + 1,
                    "Hour": hour + 1,
                    "Action": action.tolist(),
                    "Battery_Energy": info["battery_energy"],
                }
            )
            episode_reward += reward
            episode_loss += info["system_loss"]
            state = next_state
            if done:
                break

        episode_log.append(
            {
                "Episode": episode + 1,
                "Total_Reward": episode_reward,
                "Total_Loss": episode_loss,
            }
        )
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

    pd.DataFrame(step_log).to_csv("step_log_case14.csv", index=False)
    pd.DataFrame(episode_log).to_csv("episode_log_case14.csv", index=False)

# Function to start training
def run_training(case_file, episodes=360):
    """Run the entire training process."""
    env, agent = setup_environment_and_agent(case_file)
    train_agent(agent, env, episodes)

# Example usage
case_file = "case14.m"  # Replace with the actual path to case14.m
run_training(case_file, episodes=360)