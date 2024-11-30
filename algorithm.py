import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from pypower.api import runpf, loadcase, ppoption
import csv
import h5py
import datetime
import sys
from contextlib import redirect_stdout
import io
import traceback


# Utility: Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PyPowerEnv:
    """
    PyPower Environment for RL-based power flow optimization.
    """
    def __init__(self, case_file, training_data_file, testing_data_file, mode, max_steps=24):
        self.case_file = case_file
        self.training_data_file = training_data_file
        self.testing_data_file = testing_data_file
        self.max_steps = max_steps
        self.current_step = 0
        self.episodes = 0
        self.battery_energy = 2.0  # Initial SOC (MWh)
        self.mu_b = 0.98  # Charging/discharging efficiency
        self.state = None
        self.done = False
        self.mode = mode
        # Power system constraints
        self.P_bss_min, self.P_bss_max = -2, 2  # MW
        self.E_max, self.E_min = 3.6, 0.8  # MWh
        self.V_min = 1.06; 
        self.V_max = 0.94; 
        
        self.training_data = self.load_training_data()
        self.testing_data = self.load_testing_data()

    def load_training_data(self):
        try:
            training_data = loadmat(self.training_data_file)
            print("Keys in the .mat file:", training_data.keys())
            training_data = training_data["training_data"]
            return training_data
        except Exception as e:
            print(f"Error loading training data: {e}")
            raise
        
    def load_testing_data(self):
        try:
            testing_data = loadmat(self.testing_data_file)
            print("Keys in the .mat file:", testing_data.keys())
            testing_data = testing_data["test_data"]
            return testing_data
        except Exception as e:
            print(f"Error loading testing data: {e}")
            raise

    def get_mpc(self):
        """
        Construct the MATPOWER case (mpc) for the current episode using the training data.
        """
        # Select the appropriate dataset based on the mode
        if self.mode == "train":
            data = self.training_data
        elif self.mode == "test":
            data = self.testing_data
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Use 'train' or 'test'.")
        
        # Ensure row_index does not exceed the size of the data
        total_rows = data.shape[0]
        row_index = (self.episodes * self.max_steps + self.current_step) % total_rows
        current_data = data[row_index, :]
        print(f"Current row (Mode: {self.mode}, Episode {self.episodes}, Step {self.current_step}): {current_data}")

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
        self.episodes += 1
        self.state = self.get_initial_state()
        return self.state

   
    def get_initial_state(self):
        """Run power flow and return the initial state based on load demands, generation, and battery SOC."""
        try:
            # Get the mpc case
            mpc = self.get_mpc()

            # Ensure mpc data is in the correct format
            mpc["bus"] = np.array(mpc["bus"], dtype=np.float64)
            mpc["branch"] = np.array(mpc["branch"], dtype=np.float64)
            mpc["gen"] = np.array(mpc["gen"], dtype=np.float64)

            # Suppress verbose output
            ppopt = ppoption(VERBOSE=0, OUT_ALL=0)

            # Run power flow
            results, success = runpf(mpc, ppopt=ppopt)
            if not success:
                print("Power flow failed. Returning default state.")
                return np.zeros(2 * len(mpc["bus"]) + len(mpc["gen"]) + 1)  # Default state dimension

            # Extract load demands (Pload and Qload)
            P_load = results["bus"][:, 2]  # Active power demand
            Q_load = results["bus"][:, 3]  # Reactive power demand

            # Extract renewable generation (Pw and Ppv)
            P_gen = results["gen"][:, 1]  # Active power generation
            P_w = P_gen[4]  # Wind generator (assuming it's at index 4)
            P_pv = P_gen[1:2]  # Solar generators (assuming indices 1 and 2)

            # Include battery SOC
            battery_soc = self.battery_energy

            # Combine all into the state vector
            state = np.concatenate([P_load, Q_load, P_pv, [P_w], [battery_soc]])
            print(f"state shape {state.shape}")
            return state
        except Exception as e:
            state_dim = 2 * len(mpc["bus"]) + len(mpc["gen"]) + 1
            print(f"Returning fallback state of dimension {state_dim} due to error.")
            print(f"Error during power flow: {e}")
            return np.zeros(state_dim)
        
    def get_losses(self, results):
        """
        Calculate power losses from the results dictionary.

        Parameters:
            results (dict): Dictionary containing power flow results.
                            Expected to include 'branch' key with branch data.

        Returns:
            dict: Total real power losses (P_loss) and reactive power losses (Q_loss).
        """
        if 'branch' not in results:
            raise ValueError("Results dictionary must include 'branch' key.")
        
        # Extract branch data
        # Branch columns: From bus, To bus, ..., P_from, Q_from, P_to, Q_to
        branch_data = results['branch']

        # Indices for power columns (assuming MATPOWER-like format)
        P_from_col = 13  # Active power flow (from)
        Q_from_col = 14  # Reactive power flow (from)
        P_to_col = 15    # Active power flow (to)
        Q_to_col = 16    # Reactive power flow (to)

        # Calculate real and reactive power losses for each branch
        P_loss = np.sum(np.abs(branch_data[:, P_from_col] - branch_data[:, P_to_col]))
        Q_loss = np.sum(np.abs(branch_data[:, Q_from_col] - branch_data[:, Q_to_col]))

        return {
            "P_loss": P_loss,  # Total active power loss
            "Q_loss": Q_loss   # Total reactive power loss
        }

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

        self.current_step += 1

        # Update the battery energy state and calculate SOC penalty
        Pbss = action[0]  # Active power from the battery
        soc_penalty = self.update_battery(Pbss)

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
            # # Suppress PyPower output
            ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
            
            # Redirect stdout to suppress PyPower output
            with io.StringIO() as buf, redirect_stdout(buf):
                results, success = runpf(mpc, ppopt = ppopt)
            
            if not success:
                raise ValueError("Power flow did not converge.")
            
            # Power loss
            losses = self.get_losses(results)  # Use get_losses function          
            
            # Extract real power losses from the losses dictionary
            losses_real = losses["P_loss"]

            # Calculate the system loss (total real power loss)
            system_loss = losses_real

            # Assign the real power loss as a penalty for power losses
            Ploss_penalty = system_loss
            
            # Voltage penalties
            voltage = results["bus"][:, 7]  # Voltage magnitudes
            
            print("Type of voltage array:", type(voltage))
            
            voltage_penalty = np.sum((voltage < self.V_min) | (voltage > self.V_max))

            # SOC deviation penalty
            optimal_soc = (self.E_max + self.E_min) / 2
            soc_deviation_penalty = abs(self.battery_energy - optimal_soc) ** 2

            # Calculate total reward
            reward = -Ploss_penalty - soc_penalty - voltage_penalty - soc_deviation_penalty

            # Create the next state
            P_load = results["bus"][:, 2]  # Active power demands
            Q_load = results["bus"][:, 3]  # Reactive power demands
            P_gen = results["gen"][:, 1] # Active power generation
            P_w = P_gen[4]  # Wind generator
            P_pv = P_gen[1:2]  # Solar generators
            battery_soc = self.battery_energy
            next_state = np.concatenate([P_load, Q_load, P_pv, [P_w], [battery_soc]])
            print(f"Next state shape: {next_state.shape}")

            # Check if episode is complete
            self.done = self.current_step >= self.max_steps
            
            #logging
            print(f"Episode {self.episodes + 1}, Hour {self.current_step + 1}, Done: {self.done}")
            
            return next_state, reward, self.done, {
                "battery_energy": self.battery_energy,
                "system_loss": system_loss,
                "soc_penalty": soc_penalty,
                "voltage_penalty": voltage_penalty,
            }
        except Exception as e:
            print(f"Error during power flow in step function: {e}")
            traceback.print_exc()
            reward = -1.0  # Penalty for exception
            self.done = True
            return np.zeros_like(self.state), reward, self.done, {"battery_energy": self.battery_energy}
        
    def update_battery(self, P_b):
        """Update the battery SOC based on the action."""
        self.battery_energy += self.mu_b * P_b
        
        # Check for SOC violations and apply penalties
        penalty_soc = 0
        if self.battery_energy > self.E_max:
            penalty_soc += (self.battery_energy - self.E_max) ** 2  # Quadratic penalty for overcharging
            self.battery_energy = self.E_max  # Prevent exceeding max limit
        elif self.battery_energy < self.E_min:
            penalty_soc += (self.E_min - self.battery_energy) ** 2  # Quadratic penalty for over-discharging
            self.battery_energy = self.E_min  # Prevent falling below min limit
            
        """
            A penalty function focused on SOC management, designed to penalize the system when 
            SOC deviates from the optimal range, either becoming too low or exceeding a safe threshold.
            
            ^^ Quadratic penalties align well with this concept, as they impose progressively 
            higher penalties for deviations farther from the optimal range.
        """
        
        return penalty_soc  # Return SOC penalty to include in reward


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
        state = state.view(1, -1)  # Ensures correct shape (batch_size=1, input_dim=32)
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

        state_dim = (len(env.state)) if env.state is not None else 0
        action_dim = 5
        print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
        agent = PPOAgent(state_dim, action_dim)
        print(f"Agent initialized successfully.")
        return env, agent
    except Exception as e:
        print(f"Error in setup_environment_and_agent: {e}")
        raise


def train_agent(agent, env, episodes=4):
    """Train the PPO agent on the environment."""
    # Initialize logs as Python lists
    step_log = [["Date", "Episode", "Hour", "Action", "Battery_Energy", "System_Loss"]]
    episode_log = [["Date", "Episode", "Total_Reward", "Total_Loss"]]

    for episode in range(episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        episode_reward = 0
        episode_loss = 0
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Capture today's date

        for hour in range(env.max_steps):  # Iterate over the max steps (e.g., 24 for a day)
            action = agent.select_action(state)  # Get action from the agent
            next_state, reward, done, info = env.step(action)  # Take a step in the environment

            # Append step details to the log
            step_log.append([
                current_date,  # Current date
                episode + 1,  # Episode number
                hour + 1,  # Hour within the episode
                action.tolist(),  # Action taken
                info.get("battery_energy", 0),  # Battery energy state
                info.get("system_loss", 0),  # System losses
            ])

            # Update cumulative metrics
            episode_reward += reward
            episode_loss += info.get("system_loss", 0)
            state = next_state

            if done:  # Stop the episode if the environment signals completion
                break

        # Append episode summary to the log
        episode_log.append([
            current_date,  # Current date
            episode + 1,  # Episode number
            episode_reward,  # Total reward for the episode
            episode_loss,  # Total loss for the episode
        ])
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}, Total Loss = {episode_loss:.2f}")

    # Write logs to CSV files using the csv module
    with open("step_log_case_training2.csv", "w", newline="") as step_file:
        writer = csv.writer(step_file)
        writer.writerows(step_log)

    with open("episode_log_case_training2.csv", "w", newline="") as episode_file:
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
        traceback.print_exc()
        

def test_agent(agent, env, episodes=5):
    """Test the PPO agent on a separate dataset."""
    # Initialize logs as Python lists
    step_log = [["Date", "Episode", "Hour", "Action", "Reward", "Battery_Energy", "System_Loss", "Voltage_Penalty"]]
    episode_log = [["Date", "Episode", "Total_Reward"]]

    for episode in range(episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        episode_reward = 0
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Capture today's date

        for hour in range(env.max_steps):  # Iterate over the max steps (e.g., 24 for a day)
            action = agent.select_action(state)  # Get action from the agent
            next_state, reward, done, info = env.step(action)  # Take a step in the environment

            # Append step details to the log
            step_log.append([
                current_date,  # Current date
                episode + 1,  # Episode number
                hour + 1,  # Hour within the episode
                action.tolist(),  # Action taken
                reward,  # Reward received
                info.get("battery_energy", 0),  # Battery energy state
                info.get("system_loss", 0),  # System losses
                info.get("voltage_penalty", 0),  # Voltage penalty
            ])

            # Update cumulative metrics
            episode_reward += reward
            state = next_state

            if done:  # Stop the episode if the environment signals completion
                break

        # Append episode summary to the log
        episode_log.append([
            current_date,  # Current date
            episode + 1,  # Episode number
            episode_reward,  # Total reward for the episode
        ])
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

    # Write logs to CSV files using the csv module
    with open("step_log_case_testing.csv", "w", newline="") as step_file:
        writer = csv.writer(step_file)
        writer.writerows(step_log)

    with open("episode_log_case_testing.csv", "w", newline="") as episode_file:
        writer = csv.writer(episode_file)
        writer.writerows(episode_log)

    print("Testing complete. Logs saved to 'step_log_case_testing.csv' and 'episode_log_case_testing.csv'.")


def main_with_testing(case_file, testing_data_file, training_data_file, train_episodes = 1090, test_episodes=5):
    """
    Main function to set up and run the PyPowerEnv with PPO training and testing.

    Parameters:
        case_file (str): Path to the MATPOWER case file (e.g., case14.m).
        training_data_file (str): Path to the .mat file containing training data.
        testing_data_file (str): Path to the .mat file containing testing data.
        train_episodes (int): Number of episodes for training.
        test_episodes (int): Number of episodes for testing.
    """
    try:
        print(f"Initializing environment with case file: {case_file}")
        
        # # Training Phase
        # env_train = PyPowerEnv(case_file, training_data_file, testing_data_file)
        

        # agent = PPOAgent(state_dim, action_dim)
        # print("Starting training...")
        # train_agent(agent, env_train, train_episodes)

        # print("Training completed successfully!")

        # Testing Phase
        print("Starting testing...")
        mode = 'test'
        env_test = PyPowerEnv(case_file, training_data_file, testing_data_file, mode)
        state_dim = len(env_test.get_initial_state())
        action_dim = 5
        agent = PPOAgent(state_dim, action_dim)
        test_agent(agent, env_test)

        print("Testing completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()


# Example usage
if __name__ == "__main__":
    case_file = "case14mod.mat"  # Replace with the actual path to case14.m
    training_data_file = "training_data.mat"  # Replace with the actual path to training_data.mat
    testing_data_file = "test_data.mat"

    # main(case_file, training_data_file)
    main_with_testing(case_file, testing_data_file, training_data_file)
    
