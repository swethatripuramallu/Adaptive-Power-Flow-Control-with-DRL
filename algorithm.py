import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from pypower.api import runpf, loadcase
import matlab.engine
import csv
import h5py


# Utility: Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PyPowerEnv:
    """
    PyPower Environment for RL-based power flow optimization.
    """
    def __init__(self, case_file, training_data_file, max_steps=24):
        self.case_file = case_file
        self.training_data_file = training_data_file
        self.max_steps = max_steps
        self.current_step = 0
        self.episodes = 0
        self.training_data = self.load_training_data()
        self.battery_energy = 2.0  # Initial SOC (MWh)
        self.mu_b = 0.98  # Charging/discharging efficiency
        self.state = None
        self.done = False

        # Power system constraints
        self.P_bss_min, self.P_bss_max = -2, 2  # MW
        self.E_max, self.E_min = 3.6, 0.8  # MWh

    # def load_training_data(self):
    #     """Load and validate training data from a .mat file."""
    #     print(f"Loading training data from: {self.case_file}")
    #     mat_data = loadmat(self.case_file)

    #     if "training_data" not in mat_data:
    #         raise KeyError("The .mat file must contain a 'training_data' key.")

    #     data = mat_data["training_data"]
    #     if data.shape[1] != 22:
    #         raise ValueError("Training data must have exactly 22 columns.")

    #     print(f"Training data loaded. Shape: {data.shape}")
    #     return data
    # def load_training_data(self):
    #     """Load and validate training data from a .mat file."""
    #     try:
    #         print(f"Loading training data from: {self.training_data_file}")

    #         # Check the file format
    #         if self.training_data_file.endswith(".mat"):
    #             with h5py.File(self.training_data_file, "r") as mat_data:
    #                 # Extract the training data
    #                 if "training_data" not in mat_data:
    #                     raise KeyError("The .mat file must contain a 'training_data' key.")

    #                 # Load and transpose training data if needed
    #                 training_data = mat_data["training_data"][:]
    #                 if len(training_data.shape) != 3 or training_data.shape[2] != 22:
    #                     raise ValueError("Training data must have shape [days, hours, 22 columns].")

    #                 print(f"Training data loaded successfully. Shape: {training_data.shape}")
    #                 return training_data

    #         raise ValueError("File is not in the expected .mat format.")
    #     except Exception as e:
    #         print(f"Error loading training data: {e}")
    #         raise
    def load_training_data(self):
        """Load training data using MATLAB Engine."""
        try:
            print(f"Loading training data from: {self.training_data_file}")
            eng = matlab.engine.start_matlab()
            training_data = eng.load(self.training_data_file)["training_data"]
            training_data = np.array(training_data)  # Convert to NumPy array
            print(f"Training data loaded successfully. Shape: {training_data.shape}")
            eng.quit()
            return training_data
        except Exception as e:
            print(f"Error loading training data: {e}")
            raise

    def get_mpc(self):
        """
        Construct the MATPOWER case (mpc) for the current episode using the training data.
        """
        current_data = self.training_data[self.episodes % self.training_data.shape[0], :]
        print(f"Current training row: {current_data}")

        # Load the MATPOWER case
        mpc = loadcase(self.case_file)

        # Update bus data with current training data
        mpc["bus"][3, 2:4] = [current_data[4], current_data[5]]  # Pd, Qd for bus 4
        mpc["bus"][4, 2:4] = [current_data[6], current_data[7]]  # Pd, Qd for bus 5
        mpc["bus"][8, 2:4] = [current_data[8], current_data[9]]  # Pd, Qd for bus 9
        mpc["bus"][9, 2:4] = [current_data[10], current_data[11]]  # Pd, Qd for bus 10
        mpc["bus"][10, 2:4] = [current_data[12], current_data[13]]  # Pd, Qd for bus 11
        mpc["bus"][11, 2:4] = [current_data[14], current_data[15]]  # Pd, Qd for bus 12
        mpc["bus"][12, 2:4] = [current_data[16], current_data[17]]  # Pd, Qd for bus 13
        mpc["bus"][13, 2:4] = [current_data[18], current_data[19]]  # Pd, Qd for bus 14

        # Update generator data
        mpc["gen"][1, 1] = current_data[20]  # Pg for solar generation at bus 2
        mpc["gen"][2, 1] = current_data[20]  # Pg for solar generation at bus 3
        mpc["gen"][4, 1] = current_data[21]  # Pg for wind generation at bus 8

        return mpc


    def reset(self):
        """Reset the environment for a new episode."""
        print("Resetting environment...")
        self.current_step = 0
        self.done = False
        self.battery_energy = 2.0
        self.state = self.get_initial_state()
        return self.state

    def get_initial_state(self):
        """Run power flow and return initial state."""
        try:
            mpc = self.get_mpc()
            results, success = runpf(mpc)
            if not success:
                print("Power flow failed. Returning default state.")
                return np.zeros(len(mpc["bus"]))
            return results["bus"][:, 7]  # Voltage magnitudes
        except Exception as e:
            print(f"Error during power flow: {e}")
            return np.zeros(len(mpc["bus"]))

    def step(self, action):
        """
        Perform a step in the environment.

        Parameters:
            action (array): Array of 5 actions from the agent, representing:
                - action[0]: Pbss (battery active power)
                - action[1]: Qbss (battery reactive power)
                - action[2]: Qpv1 (solar reactive power for gen 2)
                - action[3]: Qpv2 (solar reactive power for gen 3)
                - action[4]: Qw (wind reactive power for gen 5)

        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Environment is done. Reset to start a new episode.")

        self.current_step += 1

        # Update the battery energy state
        Pbss = action[0]  # Active power from the battery
        self.update_battery(Pbss)

        # Fetch the current mpc
        mpc = self.get_mpc()

        # Assign actions to the mpc generator attributes
        mpc["gen"][3, 1] = Pbss  # Update Pg for battery (gen 4 in MATLAB, index 3 in Python)
        mpc["gen"][3, 2] = action[1]  # Qbss (reactive power for battery)

        mpc["gen"][1, 2] = action[2]  # Qpv1 (reactive power for solar at gen 2)
        mpc["gen"][2, 2] = action[3]  # Qpv2 (reactive power for solar at gen 3)
        mpc["gen"][4, 2] = action[4]  # Qw (reactive power for wind at gen 5)

        # Run the power flow analysis
        try:
            results, success = runpf(mpc)

            # Reward calculation (example: minimize losses)
            system_loss = np.sum(results["branch"][:, 13])  # Real power losses
            reward = -system_loss  # Negative reward for losses

            # Extract next state (e.g., bus voltage magnitudes)
            next_state = results["bus"][:, 7]

            # Check if episode is complete
            self.done = self.current_step >= self.max_steps

            # Return state, reward, done flag, and additional info
            return next_state, reward, self.done, {
                "battery_energy": self.battery_energy,
                "system_loss": system_loss,
            }
        except Exception as e:
            print(f"Error during power flow: {e}")
            reward = -1.0  # Penalty for exception
            self.done = True
            return np.zeros(len(mpc["bus"])), reward, self.done, {"battery_energy": self.battery_energy}


    def update_battery(self, P_b):
        """Update the battery SOC based on the action."""
        self.battery_energy += self.mu_b * P_b
        self.battery_energy = np.clip(self.battery_energy, self.E_min, self.E_max)



def train(env, agent, episodes=1090):
    """Train the PPO agent."""
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(env.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
            state = next_state
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")


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
    try:
        print(f"Setting up environment with case file: {case_file}")
        env = PyPowerEnv(case_file)
        print(f"Environment initialized: {env}")
        print(f"Environment state after initialization: {env.state}")
        
        if env.state is None:
            raise ValueError("The environment's state is None. Check the PyPowerEnv setup.")

        state_dim = len(env.state) if env.state is not None else 0
        action_dim = 5
        print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
        agent = PPOAgent(state_dim, action_dim)
        print(f"Agent initialized successfully.")
        return env, agent
    except Exception as e:
        print(f"Error in setup_environment_and_agent: {e}")
        raise


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

def main(case_file, training_data_file, episodes=1090):
    """
    Main function to set up and run the PyPowerEnv with PPO training.

    Parameters:
        case_file (str): Path to the MATPOWER case file (e.g., case14.m).
        training_data_file (str): Path to the .mat file containing training data.
        episodes (int): Number of episodes for training.
    """
    try:
        print(f"Initializing environment with case file: {case_file} and training data: {training_data_file}")

        # Initialize the environment and the agent
        env = PyPowerEnv(case_file, training_data_file)
        state_dim = len(env.get_initial_state())
        action_dim = 5  # 5 actions as per the step function

        # Initialize PPO agent
        agent = PPOAgent(state_dim, action_dim)
        print("Environment and PPO Agent successfully initialized.")

        # Train the agent on the environment
        train_agent(agent, env, episodes)

        print("Training completed successfully!")
    except Exception as e:
        print(f"An error occurred during the setup or training process: {e}")


# Example usage
if __name__ == "__main__":
    case_file = "case14mod.mat"  # Replace with the actual path to case14.m
    training_data_file = "training_data.mat"  # Replace with the actual path to training_data.mat

    main(case_file, training_data_file)
