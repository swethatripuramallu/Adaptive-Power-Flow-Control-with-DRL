import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from pypower.api import runpf
import csv

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
        self.episodes = 0
        self.training_data = self.load_training_data()
        self.state = None
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

    def load_training_data(self):
        """Load the training data from a .mat file."""
        if not self.case_file.endswith(".mat"):
            raise ValueError("The case file must be a .mat file.")
        
        # Load the .mat file
        mat_data = loadmat(self.case_file)
        
        if 'training_data' not in mat_data:
            raise KeyError("The .mat file must contain a 'training_data' key.")
        
        # Extract and validate the data matrix
        data_matrix = mat_data['training_data']
        if data_matrix.shape[1] != 22:
            raise ValueError("The data matrix must have 22 columns.")
        
        return data_matrix

    def get_mpc(self):
        """Convert the current training data step to a MATPOWER-compatible structure."""
        current_data = self.training_data[self.episodes % self.training_data.shape[0], :]
        mpc = {
            "baseMVA": 100,
            "bus": [
                [1, 3, current_data[4], current_data[5], 0, 0, 1, 1.06, 0, 0, 1, 1.06, 0.94],
                [2, 1, current_data[6], current_data[7], 0, 0, 1, 1.045, 0, 0, 1, 1.06, 0.94],
                [3, 1, current_data[8], current_data[9], 0, 0, 1, 1.01, 0, 0, 1, 1.06, 0.94],
                [4, 1, current_data[10], current_data[11], 0, 0, 1, 1.02, 0, 0, 1, 1.06, 0.94],
                [5, 1, current_data[12], current_data[13], 0, 0, 1, 1.056, 0, 0, 1, 1.06, 0.94],
            ],
            "gen": [
                [1, current_data[20], 0, 10, 0, 1.06, 100, 1, 332.4, 0],
                [2, current_data[21], 0, 6.87, 0, 1.045, 100, 1, 15, 0],
            ],
            "branch": [
                [1, 2, 0.01938, 0.05917, 0.0528, 9900, 0, 0, 0, 0, 1, -360, 360],
                [2, 3, 0.04699, 0.19797, 0.0438, 9900, 0, 0, 0, 0, 1, -360, 360],
            ],
            "version": "2",
        }
        return mpc

    def reset(self):
        """Reset the environment."""
        self.current_step = 0
        self.done = False
        self.battery_energy = 2.0
        self.state = self.get_initial_state()
        return self.state

    def get_initial_state(self):
        """Initialize the state."""
        try:
            print("Loading MATPOWER case file...")
            self.mpc = self.load_case_file()  # Load MATPOWER case data
            print("Running power flow...")
            results, success = runpf(self.mpc)  # Run power flow
            if not success:
                raise RuntimeError("Initial power flow did not converge")
            print("Power flow succeeded. Initial state loaded.")
            return results['bus'][:, 7]  # Return initial bus voltages (Vm column)
        except Exception as e:
            print(f"Error in get_initial_state: {e}")
            return None
    
    def step(self, action):
        """Perform one step in the environment."""
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True
            return self.state, -100.0, self.done, {"battery_energy": self.battery_energy}

        # Get next state (update using the next hour in training data)
        self.update_soc(action[3])
        self.state = self.get_initial_state()

        # Example reward calculation (replace with actual logic)
        reward = -np.random.rand()

        return self.state, reward, self.done, {"battery_energy": self.battery_energy}

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
    print(f"Environment initialized: {env}")
    print(f"Environment state: {env.state}")
    
    if env.state is None:
        raise ValueError("The environment's state is None. Check the PyPowerEnv setup.")

    state_dim = len(env.state)
    action_dim = 5
    agent = PPOAgent(state_dim, action_dim)
    print(f"Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    return env, agent

def train_agent(agent, env, episodes=1090):
    """Train the PPO agent on the environment."""
    # Initialize logs as Python lists
    step_log = [["Episode", "Hour", "Action", "Battery_Energy", "System_Loss"]]
    episode_log = [["Episode", "Total_Reward", "Total_Loss"]]

    for episode in range(episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        episode_reward = 0
        episode_loss = 0

        for hour in range(env.max_steps):  # Iterate over the max steps (e.g., 24 for a day)
            action = agent.select_action(state)  # Get action from the agent
            next_state, reward, done, info = env.step(action)  # Take a step in the environment

            # Append step details to the log
            step_log.append([
                episode + 1,  # Episode number
                hour + 1,  # Hour within the episode
                action.tolist(),  # Action taken
                info.get("battery_energy", 0),  # Battery energy state
                info.get("system_loss", 0),  # System losses
            ])

            # Update episode metrics
            episode_reward += reward
            episode_loss += info.get("system_loss", 0)
            state = next_state

            if done:  # Stop the episode if the environment signals completion
                break

        # Append episode summary to the log
        episode_log.append([
            episode + 1,  # Episode number
            episode_reward,  # Total reward for the episode
            episode_loss,  # Total loss for the episode
        ])
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}, Total Loss = {episode_loss:.2f}")

    # Write logs to CSV files using the csv module
    with open("step_log_case1.csv", "w", newline="") as step_file:
        writer = csv.writer(step_file)
        writer.writerows(step_log)

    with open("episode_log_case1.csv", "w", newline="") as episode_file:
        writer = csv.writer(episode_file)
        writer.writerows(episode_log)

    print("Training complete. Logs saved to 'step_log_case1.csv' and 'episode_log_case1.csv'.")

# Function to start training
def run_training(case_file, episodes=1090):
    """Run the entire training process."""
    env, agent = setup_environment_and_agent(case_file)
    train_agent(agent, env, episodes)

# Example usage
case_file = "training_data.mat"  # Replace with the actual path to case14.m
run_training(case_file, episodes=1090)