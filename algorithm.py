import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
import logging
from pypower.api import loadcase, runpf

# Configure logging to write to a file named training_log.log
logging.basicConfig(
    filename='training.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PyPowerEnv:
    def __init__(self, case_file, max_steps=1): #need to change max steps to 24
        """
        Initialize the PyPower environment.
        
        Parameters:
        - case_file: The MATPOWER case file (as a dictionary or filepath) to load.
        - max_steps: Maximum steps for an episode (24 steps for a 24-hour day).
        """
        self.case_file = case_file
        self.max_steps = max_steps  # Set to 24 steps to represent each hour in a day
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
            return self.case_file  # Use the provided dictionary
        return loadcase(self.case_file)  # Load the case file from filepath

    def get_initial_state(self):
        """Initialize the state with bus voltages."""
        self.mpc = self.load_case_file()  # Reload the case file
        results, success = runpf(self.mpc)
        if not success:
            raise RuntimeError("Initial power flow did not converge")
        return results['bus'][:, 7]  # Return initial bus voltages (magnitudes)

    def reset(self):
        """Reset the environment."""
        self.current_step = 0
        self.done = False
        self.battery_energy = 2.0  # Reset battery SOC at the start of each day
        self.state = self.get_initial_state()
        return self.state

    def update_soc(self, P_b):
        """
        Update the state of charge (SOC) of the battery based on active power (P_b).
        """
        # Equation (9): E(t+1) = E(t) + mu_b * P_b * time_step
        self.battery_energy += self.mu_b * P_b * 1.0  # 1-hour step

    def calculate_penalty(self):
        """
        Calculate the penalty p(t) based on SOC limits.
        """
        soc_percentage = (self.battery_energy / self.E_max) * 100  # SOC as a percentage of E_max
        penalty = 0
        
        # Equation (15): Apply penalties based on SOC levels
        if soc_percentage < 20:
            penalty -= 0.1 * (20 - soc_percentage)  # Penalty for SOC < 20%
        elif soc_percentage > 90:
            penalty -= 0.1 * (soc_percentage - 90)  # Penalty for SOC > 90%
        
        return penalty

    def step(self, action):
        penalty = 0

        # Extract battery and generator actions
        Q_pv = action[0:2]
        Q_w = action[2]
        P_bss = action[3]
        Q_bss = action[4]

        # Update actions in mpc
        self.mpc['gen'][1, 2] = Q_pv[0]
        self.mpc['gen'][2, 2] = Q_pv[1]
        self.mpc['gen'][4, 2] = Q_w
        self.mpc['gen'][3, 1] = P_bss
        self.mpc['gen'][3, 2] = Q_bss

        # Update SOC based on the battery active power
        self.update_soc(P_bss)

        # Calculate SOC penalty based on the updated battery energy
        soc_penalty = self.calculate_penalty()
        penalty += soc_penalty

        # Run power flow and calculate system losses
        try:
            bus_voltages, system_loss = self.calculate_power_flow()
        except RuntimeError:
            self.done = True
            return self.state, -100.0, self.done, {"battery_energy": self.battery_energy, "soc_penalty": penalty}

        # Voltage constraints
        for voltage in bus_voltages:
            if voltage < self.V_min or voltage > self.V_max:
                penalty -= 20  # Additional penalty for voltage violations

        # Calculate reward based on power loss and penalties
        reward = -system_loss + penalty

        # Update state and check if the episode should end
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        # Update state
        self.state = bus_voltages
        return self.state, reward, self.done, {
            "battery_energy": self.battery_energy, 
            "soc_penalty": soc_penalty,
            "system_loss": system_loss
        }


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
            nn.Tanh()  # Output in range [-1, 1]
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


# Training Loop
def train(agent, env, episodes=360): # need to change to episodes to 360
    # Open log files for storing results
    with open("training_episode_data.log", "a") as power_flow_log, open("training_step_data.log", "a") as action_state_log:
        # Write headers if files are empty
        if power_flow_log.tell() == 0:
            power_flow_log.write("Episode,Hour,Power_Loss,Reward\n")
        if action_state_log.tell() == 0:
            action_state_log.write("Episode,Hour,Action,State_of_Charge\n")

        # Loop through each episode (one day)
        for episode in range(episodes):
            state = env.reset()  # Reset environment at the beginning of each day
            done = False
            episode_reward = 0

            # 24 steps, each representing one hour of the day
            for hour in range(24):
                # Define action bounds
                action_bounds = np.array([
                    [0, 6.87],  # PV1 reactive power
                    [0, 6.87],  # PV2 reactive power
                    [0, 0.968],  # Wind reactive power
                    [-2, 2],     # Battery active power
                    [0, 0.968],  # Battery reactive power
                ])

                # Select action within bounds for each generator and battery
                action = agent.select_action(state, action_bounds)

                # Take a step in the environment
                next_state, reward, done, info = env.step(action)

                # Retrieve SOC and power loss
                soc = info["battery_energy"]
                soc_penalty = info["soc_penalty"]
                system_loss = info["system_loss"]
                
                # Log data for each hour in power_flow_data.log
                power_flow_log.write(f"{episode + 1},{hour + 1},{system_loss},{reward}\n")

                # Log action and SOC for each hour in action_state_data.log
                action_state_log.write(f"{episode + 1},{hour + 1},{action},{soc}\n")

                # Update the agent based on the reward and new state
                agent.update(state, action, reward, next_state, done)
                
                # Accumulate total reward for the episode
                episode_reward += reward
                state = next_state

            # Print the cumulative reward for the episode (one day)
            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")