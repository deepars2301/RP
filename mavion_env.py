import numpy as np
import gymnasium as gym
from gymnasium.envs.classic_control import utils
from mavion_model import Mavion # Assuming mavion_model.py is in the same directory or accessible
from rotation import quatinv, quatmul, eul2quat, deg2rad # Assuming rotation.py is accessible

class MavionEnv(gym.Env):

    def __init__(self, mavion=Mavion(), render_mode=None):

        self.mavion = mavion
        self.tau = 0.05  # seconds between state updates

        # Observation and state space
        self.pos_threshold = np.array([10.0, 10.0, 10.0]) 
        self.quat_threshold = np.array([1.0, 1.0, 1.0, 1.0]) 
        self.vel_threshold = np.finfo(np.float32).max * np.ones(3)
        self.rot_threshold = np.finfo(np.float32).max * np.ones(3)

        high_obs = np.concatenate([
                self.pos_threshold,
                self.quat_threshold,
                self.vel_threshold,
                self.rot_threshold
            ],dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high_obs, high_obs, dtype=np.float32)

        action_max = np.array([1.0, 1.0, 1.0, 1.0]) 
        action_min = np.array([0.0, 0.0, -1.0, -1.0]) 
        self.action_space = gym.spaces.Box(action_min, action_max, dtype=np.float32)
        self.action_range = np.array([mavion.MAX_ROTOR_SPD, mavion.MAX_ROTOR_SPD, mavion.MAX_FLAP_DEF, mavion.MAX_FLAP_DEF])

        self.state = np.zeros(13) 
        self.target = np.zeros(13) 
        self.obs = np.zeros(13)   
        self.last_applied_action_u = np.zeros_like(self.action_range) 

        # --- Reward Configuration Parameters ---
        self.REWARD_ALIVE = 0.2                
        self.PENALTY_BOUNDARY_VIOLATION = -20.0 
        self.REWARD_MAX_STEPS_SURVIVED = 20.0     
        
        # Characteristic error values for linear reward scaling (reward = 1 at error=0, 0 at error=MAX)
        self.POS_ERROR_MAX_REWARD_SCALE = 0.1  # Max total 3D position error (m) for reward scaling
        self.ALTITUDE_ERROR_MAX_REWARD_SCALE = 0.05 # Max Z-axis (altitude) error (m) for its specific reward scaling
        self.ANGLE_ERROR_MAX_REWARD_SCALE = np.pi / 2 # Max angle error (rad, 90 deg) for reward scaling
        self.VEL_ERROR_MAX_REWARD_SCALE = 0.2   # Max velocity error (m/s) for reward scaling
        self.ROT_ERROR_MAX_REWARD_SCALE = np.pi / 2 # Max angular velocity error (rad/s) for reward scaling

        # --- Rotor Speed Bonus Parameters ---
        self.ROTOR_SPEED_THRESHOLD_BONUS = 530.0 # rad/s
        self.REWARD_ROTOR_SPEED_BONUS = 0.1     # Small positive reward if both rotors are above threshold
        
        self.current_step_in_episode = 0
        self.MAX_EPISODE_STEPS = 1000
        
        # --- Boundary Failure Thresholds (NED coordinates: positive Z is Down) ---
        self.Z_TOO_HIGH_THRESHOLD = -5.10 # Terminate if z <= -5.10m (goes above -5.0 by 0.1m)
        self.Z_GROUND_THRESHOLD = -4.95   # Terminate if z >= -4.95m (goes below -5.0 by 0.05m)   
        self.XY_BOUNDARY_THRESHOLD = 0.1 


    def observation(self):
        obs = np.zeros(13)
        obs[0:3] = self.state[0:3] - self.target[0:3] 
        obs[3:7] = quatmul(quatinv(self.target[3:7]), self.state[3:7]) 
        obs[7:10] = self.state[7:10] - self.target[7:10] 
        obs[10:13] = self.state[10:13] - self.target[10:13] 
        return obs

    def reward(self):
        current_s = self.state
        target_s = self.target

        pos = current_s[0:3]
        quat = current_s[3:7]
        vel = current_s[7:10]
        rot = current_s[10:13]

        target_pos = target_s[0:3]   
        target_att = target_s[3:7] 
        target_vel = target_s[7:10]    
        target_rot = target_s[10:13]   

        log_info = {} 

        # --- Calculate Errors ---
        err_pos_total = np.linalg.norm(pos - target_pos) # Total 3D position error
        log_info['error_pos_total_m'] = err_pos_total

        err_alt = np.abs(pos[2] - target_pos[2]) # Specific Z-axis (altitude) error
        log_info['error_altitude_m'] = err_alt

        abs_dot = np.clip(np.abs(np.sum(quat * target_att)), -1.0, 1.0)
        err_att_rad = 2 * np.arccos(abs_dot)
        log_info['error_angle_rad'] = err_att_rad
        log_info['error_angle_deg'] = np.rad2deg(err_att_rad)

        err_vel_mag = np.linalg.norm(vel - target_vel)
        log_info['error_vel_magnitude_mps'] = err_vel_mag

        err_rot_mag = np.linalg.norm(rot - target_rot)
        log_info['error_rot_magnitude_radps'] = err_rot_mag

        # Determine if terminated by boundary conditions
        terminated_by_too_high = bool(pos[2] <= self.Z_TOO_HIGH_THRESHOLD) 
        terminated_by_ground_hit = bool(pos[2] >= self.Z_GROUND_THRESHOLD)
        terminated_by_x_boundary = bool(np.abs(pos[0]) >= self.XY_BOUNDARY_THRESHOLD)
        terminated_by_y_boundary = bool(np.abs(pos[1]) >= self.XY_BOUNDARY_THRESHOLD)

        step_reward = self.REWARD_ALIVE 
        rotor_bonus_applied = 0.0

        # --- Linearly scaled rewards: 1.0 at zero error, 0.0 at MAX_ERROR_FOR_REWARD ---
        
        # Reward for total 3D position error
        r_pos_total_lin = np.maximum(0.0, 1.0 - (err_pos_total / self.POS_ERROR_MAX_REWARD_SCALE))
        step_reward += r_pos_total_lin
        log_info['reward_lin_pos_total'] = r_pos_total_lin

        # Specific reward for Z-axis (altitude) error
        r_alt_lin = np.maximum(0.0, 1.0 - (err_alt / self.ALTITUDE_ERROR_MAX_REWARD_SCALE))
        step_reward += r_alt_lin
        log_info['reward_lin_altitude'] = r_alt_lin

        # Reward for angular error
        r_att_lin = np.maximum(0.0, 1.0 - (err_att_rad / self.ANGLE_ERROR_MAX_REWARD_SCALE))
        step_reward += r_att_lin
        log_info['reward_lin_angle'] = r_att_lin

        # Reward for linear velocity error
        r_vel_lin = np.maximum(0.0, 1.0 - (err_vel_mag / self.VEL_ERROR_MAX_REWARD_SCALE))
        step_reward += r_vel_lin
        log_info['reward_lin_vel'] = r_vel_lin

        # Reward for angular velocity error
        r_rot_lin = np.maximum(0.0, 1.0 - (err_rot_mag / self.ROT_ERROR_MAX_REWARD_SCALE))
        step_reward += r_rot_lin
        log_info['reward_lin_rot'] = r_rot_lin

        # Add bonus for sufficient rotor speeds (only if not terminated by other conditions yet)
        if not (terminated_by_too_high or terminated_by_ground_hit or terminated_by_x_boundary or terminated_by_y_boundary):
            rotor_speed_1 = self.last_applied_action_u[0]
            rotor_speed_2 = self.last_applied_action_u[1]
            if rotor_speed_1 > self.ROTOR_SPEED_THRESHOLD_BONUS and \
               rotor_speed_2 > self.ROTOR_SPEED_THRESHOLD_BONUS:
                step_reward += self.REWARD_ROTOR_SPEED_BONUS
                rotor_bonus_applied = self.REWARD_ROTOR_SPEED_BONUS
        
        log_info['reward_component_rotor_bonus'] = rotor_bonus_applied
        log_info['subtotal_step_reward_before_termination_mods'] = step_reward
        
        log_info['actual_rotor_speed_1'] = self.last_applied_action_u[0] 
        log_info['actual_rotor_speed_2'] = self.last_applied_action_u[1]
        
        return step_reward, terminated_by_too_high, terminated_by_ground_hit, terminated_by_x_boundary, terminated_by_y_boundary, log_info


    def step(self, action):
        self.current_step_in_episode += 1
        infos = {} 

        infos["raw_action"] = np.copy(action) 

        u = np.zeros_like(self.action_range)
        u[0] = action[0] * self.mavion.MAX_ROTOR_SPD 
        u[1] = action[1] * self.mavion.MAX_ROTOR_SPD 
        u[2] = action[2] * self.mavion.MAX_FLAP_DEF  
        u[3] = action[3] * self.mavion.MAX_FLAP_DEF  
        infos["scaled_action_u"] = np.copy(u)
        self.last_applied_action_u = u 

        self.state  = self.mavion.step(self.state, u, np.zeros(3), self.tau) 
        self.obs = self.observation() 
        
        step_reward_calculated, terminated_by_too_high, terminated_by_ground_hit, \
            terminated_by_x_boundary, terminated_by_y_boundary, reward_log_info = self.reward()
        infos.update(reward_log_info) 
        
        is_truncated_by_time = bool(self.current_step_in_episode >= self.MAX_EPISODE_STEPS)
        
        terminated_by_env_boundaries = (terminated_by_too_high or 
                                        terminated_by_ground_hit or 
                                        terminated_by_x_boundary or 
                                        terminated_by_y_boundary)

        final_reward_for_this_step = step_reward_calculated 

        if terminated_by_env_boundaries: 
            final_reward_for_this_step = self.PENALTY_BOUNDARY_VIOLATION
        elif is_truncated_by_time: 
            final_reward_for_this_step = self.REWARD_MAX_STEPS_SURVIVED 
        
        infos["final_reward_value"] = final_reward_for_this_step 
        infos["terminated_by_too_high"] = terminated_by_too_high
        infos["terminated_by_ground_hit"] = terminated_by_ground_hit
        infos["terminated_by_x_boundary"] = terminated_by_x_boundary
        infos["terminated_by_y_boundary"] = terminated_by_y_boundary
        infos["is_truncated_by_time"] = is_truncated_by_time 

        infos["current_state_full"] = np.copy(self.state)
        infos["pos_z_m"] = self.state[2] 
        infos["pos_x_m"] = self.state[0]
        infos["pos_y_m"] = self.state[1]
        
        return self.obs, final_reward_for_this_step, terminated_by_env_boundaries, is_truncated_by_time, infos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.current_step_in_episode = 0 

        target_altitude = -5.0 
        target_quat_pitch_90 = np.array([np.cos(np.pi/4), 0.0, np.sin(np.pi/4), 0.0]) 

        self.target = np.array([
            0.0, 0.0, target_altitude,   
            target_quat_pitch_90[0], target_quat_pitch_90[1], 
            target_quat_pitch_90[2], target_quat_pitch_90[3], 
            0.0, 0.0, 0.0,               
            0.0, 0.0, 0.0                
        ])

        low_bound, high_bound = utils.maybe_parse_reset_bounds(options, -0.05, 0.05) 
        pos_perturbation = self.np_random.uniform(low=low_bound, high=high_bound, size=(3,))
        
        initial_state = np.copy(self.target)
        initial_state[0:3] += pos_perturbation 
        
        if initial_state[2] <= self.Z_TOO_HIGH_THRESHOLD: 
            initial_state[2] = self.Z_TOO_HIGH_THRESHOLD + 0.01 # Adjusted to ensure it's not exactly on threshold 
        elif initial_state[2] >= self.Z_GROUND_THRESHOLD: 
            initial_state[2] = self.Z_GROUND_THRESHOLD - 0.01 # Adjusted
        
        if np.abs(initial_state[0]) >= self.XY_BOUNDARY_THRESHOLD:
            initial_state[0] = np.sign(initial_state[0]) * (self.XY_BOUNDARY_THRESHOLD - 0.01) # Adjusted
        if np.abs(initial_state[1]) >= self.XY_BOUNDARY_THRESHOLD:
            initial_state[1] = np.sign(initial_state[1]) * (self.XY_BOUNDARY_THRESHOLD - 0.01) # Adjusted


        self.state = initial_state
        self.obs = self.observation() 
        
        initial_infos = {} 
        initial_infos["initial_state_full"] = np.copy(self.state)
        initial_infos["initial_observation_full"] = np.copy(self.obs)
        initial_infos["target_state_full"] = np.copy(self.target)
        
        self.last_applied_action_u = np.zeros_like(self.action_range) 
        
        return self.obs, initial_infos

    def render(self):
        pass 

    def close(self):
        pass 

# Example usage block
if __name__ == '__main__':
    try:
        mavion_model_instance = Mavion()
    except FileNotFoundError:
        print("Error: mavion.yaml not found. Please ensure it's in the correct directory.")
        print("Skipping MavionEnv example usage.")
        exit()

    env = MavionEnv(mavion=mavion_model_instance)
    
    print("--- Testing reset() ---")
    obs, info_reset = env.reset() 
    print(f"ALTITUDE_ERROR_MAX_REWARD_SCALE: {env.ALTITUDE_ERROR_MAX_REWARD_SCALE:.2f}m")
    print(f"POS_ERROR_MAX_REWARD_SCALE: {env.POS_ERROR_MAX_REWARD_SCALE:.2f}m")

    print("-" * 30)

    print("--- Testing reward() logic with linear scaling ---")
    
    # Test Case 1: Perfect state
    env.reset()
    env.state = np.copy(env.target) 
    env.last_applied_action_u = np.array([550.0, 550.0, 0.0, 0.0]) # Rotors above threshold for bonus
    print(f"\nTest Case 1: Perfect state")
    step_r, _, _, _, _, r_info = env.reward() 
    print(f"  Subtotal Step Reward: {r_info.get('subtotal_step_reward_before_termination_mods'):.4f}") 
    # Expected: REWARD_ALIVE + (1_pos + 1_alt + 1_angle + 1_vel + 1_rot) + ROTOR_BONUS
    # Expected: 0.2 + 5.0 + 0.1 = 5.3
    print(f"  reward_lin_pos_total: {r_info.get('reward_lin_pos_total'):.4f}")
    print(f"  reward_lin_altitude: {r_info.get('reward_lin_altitude'):.4f}")
    print(f"  reward_lin_angle: {r_info.get('reward_lin_angle'):.4f}") 
    print(f"  reward_lin_vel: {r_info.get('reward_lin_vel'):.4f}")     
    print(f"  reward_lin_rot: {r_info.get('reward_lin_rot'):.4f}")
    print(f"  reward_component_rotor_bonus: {r_info.get('reward_component_rotor_bonus'):.4f}")


    # Test Case 2: Altitude error at half of its MAX_REWARD_SCALE, other errors zero
    env.reset()
    env.state = np.copy(env.target)
    env.state[2] += env.ALTITUDE_ERROR_MAX_REWARD_SCALE / 2.0 # err_alt = MAX_SCALE / 2
    # This will also affect err_pos_total
    env.last_applied_action_u = np.array([550.0, 550.0, 0.0, 0.0])
    print(f"\nTest Case 2: Altitude error at 50% of its max scale for reward, other errors zero (pos_total will also have error)")
    step_r, _, _, _, _, r_info = env.reward()
    print(f"  Subtotal Step Reward: {r_info.get('subtotal_step_reward_before_termination_mods'):.4f}") 
    print(f"  reward_lin_altitude: {r_info.get('reward_lin_altitude'):.4f}") # Expect 0.5
    print(f"  reward_lin_pos_total: {r_info.get('reward_lin_pos_total'):.4f}") # Will be < 1.0 due to z-error
    print(f"  error_altitude_m: {r_info.get('error_altitude_m'):.4f}")
    print(f"  error_pos_total_m: {r_info.get('error_pos_total_m'):.4f}")


    env.close()
