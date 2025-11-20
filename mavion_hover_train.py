import numpy as np
from stable_baselines3 import SAC, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from mavion_env import MavionEnv
from rotation import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from functools import partial
import json

class MaxRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MaxRewardCallback, self).__init__(verbose)
        self.max_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        self.distance_lengths= []
        self.distance_pos = []
        self.distance_z= []
        self.distance_quat= []
        self.distance_vel= []
        self.distance_rot= []
        self.counter= 0
        # Clear the file contents on initialization so that each run starts fresh.
        with open("training_data.json", "w") as f:
            f.write("")
    
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        
        for info in infos:
            self.counter += 1
            if self.counter%100 == 0:
                with open("training_data.json", "a") as f:
                    json.dump(info, f, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
                    f.write("\n")
            if "z_dist" in info:
                z_pos_val = info["z_dist"]
                self.distance_z.append(z_pos_val)

                mean_z= np.mean(self.distance_z)
                # Record it in TensorBoard logs
                self.logger.record("rollout/mean_z", mean_z)
                # Or just print it to the console:
                # print(f"dist: {dist_val}")
                # Check if the Monitor wrapper has added episode info
            if "error_quat" in info:
                quat_val = info["error_quat"]
                self.distance_quat.append(quat_val)

                mean_quat= np.mean(self.distance_quat)
                # Record it in TensorBoard logs
                self.logger.record("rollout/mean_quat", mean_quat)
                # Or just print it to the console:
                # print(f"dist: {dist_val}")
                # Check if the Monitor wrapper has added episode info
            if "error_vel" in info:
                vel_val = info["error_vel"]
                self.distance_vel.append(vel_val)

                mean_vel= np.mean(self.distance_vel)
                # Record it in TensorBoard logs
                self.logger.record("rollout/mean_vel", mean_vel)
                # Or just print it to the console:
                # print(f"dist: {dist_val}")
                # Check if the Monitor wrapper has added episode info
            if "error_rot" in info:
                rot_val = info["error_rot"]
                self.distance_rot.append(rot_val)

                mean_rot= np.mean(self.distance_rot)
                # Record it in TensorBoard logs
                self.logger.record("rollout/mean_rot", mean_rot)
                # Or just print it to the console:
                # print(f"dist: {dist_val}")
                # Check if the Monitor wrapper has added episode info
            if "error_pos" in info:
                pos_val = info["error_pos"]
                self.distance_pos.append(pos_val)

                mean_pos= np.mean(self.distance_pos)
                # Record it in TensorBoard logs
                self.logger.record("rollout/mean_pos", mean_pos)
                # Or just print it to the console:
                # print(f"dist: {dist_val}")
                # Check if the Monitor wrapper has added episode info
            if "dist" in info:
                dist_val = info["dist"]
                self.distance_lengths.append(dist_val)

                mean_distance = np.mean(self.distance_lengths)
                # Record it in TensorBoard logs
                self.logger.record("rollout/mean_dist", mean_distance)
                # Or just print it to the console:
                # print(f"dist: {dist_val}")
                # Check if the Monitor wrapper has added episode info
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                
                # Update our episode statistics
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                
                # Update max reward if needed
                if ep_reward > self.max_reward:
                    self.max_reward = ep_reward

                # Compute mean episode reward and mean episode length
                mean_reward = np.mean(self.episode_rewards)
                mean_length = np.mean(self.episode_lengths)
                
                # Log the metrics
                self.logger.record("rollout/max_reward", self.max_reward)
                self.logger.record("rollout/mean_reward", mean_reward)
                self.logger.record("rollout/mean_length", mean_length)     
        return True


# Define a factory function to create a new environment instance.
def make_env():
    def _init():
        pos_ini = np.array([0, 0, -5])
        eul_ini = np.array([0, 90, 0]) * deg2rad
        quat_ini = eul2quat(eul_ini)

        pos_target = np.array([0, 0, -5])
        eul_target = np.array([0, 90, 0]) * deg2rad
        quat_target = eul2quat(eul_target)

        env = Monitor(MavionEnv())
        env.state = np.concatenate([pos_ini, quat_ini, np.zeros(3), np.zeros(3)])
        env.target = np.concatenate([pos_target, quat_target, np.zeros(3), np.zeros(3)])
        return env
    return _init

if __name__ == "__main__":
    # Number of parallel environments.
    num_envs = 15
    # Create a list of environment creation functions.
    env_fns = [make_env() for _ in range(num_envs)]
    # Wrap the list in a DummyVecEnv to run them in parallel.
    # Replace DummyVecEnv with SubprocVecEnv for real parallelism
    vec_env = DummyVecEnv(env_fns)


    # Create the SAC model with the vectorized environment.
    model = TD3(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1
    )
    #model = SAC.load("Mavion_Own/Training Data/SAC_Hover_5.zip", env=vec_env)
    max_reward_callback = MaxRewardCallback(verbose=1)
    model.learn(total_timesteps=30, log_interval=1, progress_bar=False, callback= max_reward_callback)
    model.save("Training Data/TD3_Base")
    
    #callback=max_reward_callback