import numpy as np
from stable_baselines3 import SAC
# Import SubprocVecEnv for parallel environments
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import json
import os

# Assuming mavion_env.py (containing MavionEnv) and rotation.py are in the same directory or PYTHONPATH
# Ensure this import matches your local filename for the environment.
from mavion_env import MavionEnv # Make sure mavion_env.py contains the latest MavionEnv
from rotation import eul2quat, deg2rad # Make sure rotation.py has these

class InfoSavingCallback(BaseCallback):
    """
    A custom callback to save mean episode reward and length to a JSON file
    every N failed episodes (ground hits), and log various metrics to TensorBoard.
    """
    def __init__(self, verbose=0, save_stats_on_nth_failure=10, json_log_filename="training_summary_stats.json"):
        super(InfoSavingCallback, self).__init__(verbose)
        # For general episode statistics logged to TensorBoard
        self.max_reward_overall = -np.inf
        self.all_episode_rewards = [] # Stores rewards of all completed episodes for running means
        self.all_episode_lengths = [] # Stores lengths of all completed episodes for running means
        
        # For logging specific step-wise metrics from 'infos' to TensorBoard (running averages)
        self.metric_history = {
            # These keys should match what's available in the 'infos' dict from MavionEnv
            "error_pos_total_m": [],
            "error_angle_deg": [],
            "error_vel_magnitude_mps": [],
            "error_rot_magnitude_radps": [],
            "pos_z_m": [], # Actual altitude
            "reward_exp_pos": [],
            "reward_exp_angle": [],
            "reward_exp_vel": [],
            "reward_exp_rot": []
        }
        self.log_freq_steps = 100 # How often to log running averages of step metrics to TensorBoard

        # For saving summary stats to JSON based on failed episodes
        self.save_stats_on_nth_failure = save_stats_on_nth_failure
        self.ground_hit_counter = 0 # Counts episodes ending due to ground hit
        self.json_log_filename = json_log_filename

        # Initialize/clear the JSON log file
        # Stores a list of summary statistic objects
        with open(self.json_log_filename, "w") as f:
            json.dump([], f) 
        if self.verbose > 0:
            print(f"Callback: Initialized JSON log file for summary stats: {self.json_log_filename}")

    def _on_step(self) -> bool:
        infos_per_env = self.locals.get("infos", [{} for _ in range(self.training_env.num_envs)])
        dones_per_env = self.locals.get("dones", [False for _ in range(self.training_env.num_envs)])

        for i in range(self.training_env.num_envs):
            info_dict = infos_per_env[i]
            
            # --- Log per-step metrics from info_dict to TensorBoard (running averages) ---
            for key in self.metric_history.keys():
                if key in info_dict:
                    self.metric_history[key].append(info_dict[key])
            
            if self.num_timesteps % self.log_freq_steps == 0 and self.num_timesteps > 0:
                if self.verbose > 1:
                    print(f"Callback: Logging step metrics at timestep {self.num_timesteps}")
                for key, values in self.metric_history.items():
                    if values: 
                        mean_val = np.mean(values[-1000:]) # Log mean of recent values
                        self.logger.record(f"metrics_step_mean/{key}", mean_val)

            # --- Process ended episodes ---
            if dones_per_env[i]: 
                if self.verbose > 1:
                    print(f"Callback: Episode ended in env {i} at timestep {self.num_timesteps}. Info: {info_dict}")
                
                # Check for specific termination reasons from MavionEnv
                was_ground_hit = info_dict.get("terminated_by_ground_hit", False)
                # You can add other termination flags here if needed, e.g., "terminated_by_too_high"
                
                if self.verbose > 1:
                    print(f"Callback: Env {i} - terminated_by_ground_hit: {was_ground_hit}")

                if was_ground_hit: # Only trigger JSON save on ground hits
                    self.ground_hit_counter += 1
                    if self.verbose > 0:
                        print(f"Callback: Env {i} - Ground Hit! Ground hit count: {self.ground_hit_counter}.")

                    # Check if it's time to save summary stats to JSON
                    if self.ground_hit_counter > 0 and self.ground_hit_counter % self.save_stats_on_nth_failure == 0:
                        if self.verbose > 0:
                            print(f"Callback: Triggering JSON save due to {self.ground_hit_counter} ground hits.")
                        
                        current_mean_reward = np.mean(self.all_episode_rewards[-100:]) if self.all_episode_rewards else None
                        current_mean_length = np.mean(self.all_episode_lengths[-100:]) if self.all_episode_lengths else None

                        if current_mean_reward is not None and current_mean_length is not None:
                            log_entry = {
                                "timestep": self.num_timesteps,
                                "mean_episode_reward_last_100": current_mean_reward,
                                "mean_episode_length_last_100": current_mean_length,
                                "triggering_ground_hit_count": self.ground_hit_counter
                            }
                            
                            try:
                                with open(self.json_log_filename, "r") as f:
                                    existing_data = json.load(f)
                            except (FileNotFoundError, json.JSONDecodeError):
                                existing_data = [] 
                            
                            existing_data.append(log_entry)
                            
                            try:
                                with open(self.json_log_filename, "w") as f:
                                    json.dump(existing_data, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else str(o))
                                if self.verbose > 0:
                                    print(f"Callback: Successfully saved summary stats to {self.json_log_filename}.")
                            except Exception as e:
                                if self.verbose > 0:
                                    print(f"Callback: CRITICAL ERROR writing summary stats to JSON file: {e}")
                        elif self.verbose > 0:
                            print(f"Callback: Not enough episode data to log mean reward/length for JSON.")
                
                # --- Standard episode logging (reward, length) using Monitor's info ---
                if "episode" in info_dict:
                    ep_reward = info_dict["episode"]["r"]
                    ep_length = info_dict["episode"]["l"]
                    
                    self.all_episode_rewards.append(ep_reward)
                    self.all_episode_lengths.append(ep_length)
                    
                    if ep_reward > self.max_reward_overall:
                        self.max_reward_overall = ep_reward
                    
                    if self.all_episode_rewards: 
                         self.logger.record("rollout/ep_rew_mean", np.mean(self.all_episode_rewards[-100:]))
                         self.logger.record("rollout/ep_len_mean", np.mean(self.all_episode_lengths[-100:]))
                    self.logger.record("rollout/max_reward_overall", self.max_reward_overall)
        return True


def make_env_factory(rank, seed=0, env_config=None):
    """
    Utility function for multiprocessed env.
    """
    if env_config is None:
        env_config = {}
    def _init():
        env = MavionEnv(**env_config) 
        env = Monitor(env) 
        env.reset(seed=seed + rank) 
        return env
    return _init

def linear_schedule(initial_value: float, final_value: float):
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :param final_value: Final learning rate.
    :return: schedule that computes current learning rate depending on progress.
    """
    def func(progress_remaining: float) -> float:
        """
        Progress_remaining will go from 1.0 to 0.0 over the course of training.
        """
        return final_value + progress_remaining * (initial_value - final_value)
    return func

if __name__ == "__main__":
    # --- Configuration ---
    NUM_ENVS = 23 
    TOTAL_TIMESTEPS = 1000000 
    MODEL_NAME = "SAC_Mavion_Hover_AdaptiveLR" # Base name for saving and loading
    MODEL_SAVE_PATH = f"Training_Data/{MODEL_NAME}" 
    MODEL_LOAD_PATH = f"Training_Data/{MODEL_NAME}.zip" # Path to load the model from

    JSON_LOG_FILENAME = f"training_summary_stats_{MODEL_NAME}.json" 
    SAVE_STATS_ON_NTH_FAILURE = 10 

    # Learning Rate Schedule Parameters
    INITIAL_LR = 3e-4 
    FINAL_LR = 1e-5   

    # Create a list of environment creation functions.
    if NUM_ENVS > 1:
        vec_env = SubprocVecEnv([make_env_factory(i, env_config=None) for i in range(NUM_ENVS)])
    else: 
        vec_env = DummyVecEnv([make_env_factory(0, env_config=None)])


    # --- Model and Training ---
    policy_kwargs = dict(net_arch=[256, 256])
    lr_schedule = linear_schedule(INITIAL_LR, FINAL_LR)

    # Check if a pre-trained model exists
    if os.path.exists(MODEL_LOAD_PATH):
        print(f"Loading pre-trained model from: {MODEL_LOAD_PATH}")
        model = SAC.load(
            MODEL_LOAD_PATH, 
            env=vec_env,
            learning_rate=lr_schedule, # Re-apply schedule, SB3 handles progress
            # Other parameters like buffer_size, learning_starts are part of the loaded model's state
            # but you can override them if necessary. For continuing training,
            # it's often best to let them be as they were or re-evaluate.
            # For simplicity, we are mainly focusing on re-applying the learning rate schedule.
            custom_objects={"learning_rate": lr_schedule} # Helps ensure schedule is used
        )
        # If you want to reset the num_timesteps of the loaded model to 0 for the new learning session:
        # model.set_num_timesteps(0) # Uncomment if you want to "reset" the training step count for the new session
        print(f"Model loaded. Current timesteps: {model.num_timesteps}")
    else:
        print(f"No pre-trained model found at {MODEL_LOAD_PATH}. Creating a new model.")
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=lr_schedule,  
            buffer_size=int(1e6),       
            learning_starts=25000,      
            batch_size=256,             
            gamma=0.99,                 
            tau=0.005,                  
            ent_coef='auto',            
            policy_kwargs=policy_kwargs, 
            gradient_steps=1,           
            target_update_interval=1,   
            verbose=1,
            tensorboard_log=f"./{MODEL_NAME}_tensorboard/" 
        )
    
    info_saving_callback = InfoSavingCallback(
        verbose=1, 
        save_stats_on_nth_failure=SAVE_STATS_ON_NTH_FAILURE,
        json_log_filename=JSON_LOG_FILENAME
    )

    print(f"Starting/Continuing training with {NUM_ENVS} parallel environments for {TOTAL_TIMESTEPS} total timesteps (considering loaded model's steps).")
    print(f"Using adaptive learning rate: Initial={INITIAL_LR}, Final={FINAL_LR}")
    print(f"Summary stats will be saved to {JSON_LOG_FILENAME} every {SAVE_STATS_ON_NTH_FAILURE} ground hits.")
    
    # The total_timesteps for model.learn() is the overall number of steps for this session.
    # If loading a model, it continues from model.num_timesteps up to TOTAL_TIMESTEPS.
    # If you want to train for an *additional* TOTAL_TIMESTEPS, you'd adjust this.
    # For now, assuming TOTAL_TIMESTEPS is the grand total.
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=1000, 
        progress_bar=True,
        callback=info_saving_callback,
        reset_num_timesteps=False # Important: Set to False to continue step count from loaded model
    )
    
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH) # Saves as MODEL_SAVE_PATH.zip
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}.zip")

    vec_env.close()
