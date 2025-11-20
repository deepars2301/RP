import numpy as np
import matplotlib.pyplot as plt
# Assuming mavion_model.py (containing the Mavion class) is in the same directory.
from mavion_model import Mavion
# Assuming rotation.py (containing the eul2quat function) is in the same directory.
from rotation import eul2quat
from mavion_env import MavionEnv
from stable_baselines3 import TD3, SAC

mav_env = MavionEnv()

def quat_to_euler_angles(q):
    """
    Convert a quaternion into Tait-Bryan Euler angles (roll, pitch, yaw).
    Assumes ZYX rotation sequence (yaw, pitch, roll).
    q: quaternion [qw, qx, qy, qz]
    Returns: [roll, pitch, yaw] in radians
    """
    qw, qx, qy, qz = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def run_mavion_simulation_timesteps_with_plotting():
    """
    This function initializes the Mavion drone, defines an initial state,
    and wind. It then simulates the drone's movement for a fixed number
    of timesteps using control inputs from an RL agent (via a placeholder function),
    stores the history, and plots key state variables.
    """
    try:
        # 1. Initialize the Mavion drone instance.
        drone = Mavion(file="mavion.yaml")
        print("Mavion drone initialized successfully using 'mavion.yaml'.\n")

        # 2. Define the initial state vector 'x_current' (13 elements).
        pos_ned_initial = np.array([0.0, 0.0, -5.0])      # Initial position (North, East, Down)
        quat_ned_to_body_initial = eul2quat([0.0, 90.0*np.pi/180, 0.0]) # Initial attitude (roll, pitch, yaw) -> level
        vel_ned_initial = np.array([0.0, 0.0, 0.0])       # Initial linear velocity in NED
        rot_body_initial = np.array([0.0, 0.0, 0.0])      # Initial angular velocity in body frame

        x_current = np.concatenate([pos_ned_initial, quat_ned_to_body_initial, vel_ned_initial, rot_body_initial])
        print(f"Initial state vector (x_current) at t=0:\n{x_current}\n")

        # 3. Control input 'u_input' will be determined by the RL agent at each step.
        # No constant u_input is defined here.

        # 4. Define constant wind vector 'w_input' (3 elements) in NED frame.
        w_input = np.array([0.0, 0.0, 0.0]) # Example: No wind
        # w_input = np.array([2.0, -1.0, 0.0]) # Example: Wind from NW
        print(f"Constant wind vector (w_input):\n{w_input}\n")

        # 5. Simulation parameters
        num_timesteps = 20 # Increased for a more meaningful plot with RL agent
        dt = 0.05           # Timestep duration in seconds (50 Hz)
        total_time = num_timesteps * dt
        
        print(f"Simulating for {num_timesteps} timesteps (Total: {total_time:.2f}s) with dt = {dt}s...\n")
        
        # Lists to store history for plotting
        time_history = [0.0]
        pos_ned_history = [x_current[0:3]]
        quat_history = [x_current[3:7]]
        vel_ned_history = [x_current[7:10]]
        rot_body_history = [x_current[10:13]]
        euler_angles_history = [quat_to_euler_angles(x_current[3:7])]
        control_action_history = [] # To store actions from RL agent

        mav_agent = SAC.load("TrainingData_TD3\SAC_hover_Total4MSteps.zip", env=mav_env)
        mav_env.reset()
        x0 = mav_env.state

        print("--- Simulation Start ---")
        current_time_val = 0.0
        for i in range(num_timesteps):
            # Get control action from the RL agent
            act, _ = mav_agent.predict(x0, deterministic=True)
            u = act * mav_env.action_range
            
            # Step the simulation
            x_next = drone.step(x_current, u, w_input, dt)
            x_current = x_next
            current_time_val += dt
            
            # Store history
            time_history.append(current_time_val)
            pos_ned_history.append(x_current[0:3])
            quat_history.append(x_current[3:7])
            vel_ned_history.append(x_current[7:10])
            rot_body_history.append(x_current[10:13])
            euler_angles_history.append(quat_to_euler_angles(x_current[3:7]))
            control_action_history.append(u)


            if (i + 1) % 25 == 0: # Print status every 25 steps
                print(f"  Completed step {i+1}/{num_timesteps} (t = {current_time_val:.2f}s)")


        print("\n--- Simulation End ---")

        # Convert history lists to numpy arrays for easier slicing
        pos_ned_history = np.array(pos_ned_history)
        vel_ned_history = np.array(vel_ned_history)
        euler_angles_history = np.array(euler_angles_history) * (180.0 / np.pi) # Convert to degrees for plotting
        rot_body_history = np.array(rot_body_history) * (180.0 / np.pi) # Convert to degrees/s for plotting
        control_action_history = np.array(control_action_history)


        # 6. Plotting the results
        fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True) # Added one more subplot for controls
        fig.suptitle('Mavion Drone Simulation Results with RL Agent Controls', fontsize=16)

        # Plot Position (NED)
        axs[0].plot(time_history, pos_ned_history[:, 0], label='North (m)')
        axs[0].plot(time_history, pos_ned_history[:, 1], label='East (m)')
        axs[0].plot(time_history, pos_ned_history[:, 2], label='Down (m)')
        axs[0].set_ylabel('Position (NED) [m]')
        axs[0].legend()
        axs[0].grid(True)

        # Plot Velocity (NED)
        axs[1].plot(time_history, vel_ned_history[:, 0], label='V_north (m/s)')
        axs[1].plot(time_history, vel_ned_history[:, 1], label='V_east (m/s)')
        axs[1].plot(time_history, vel_ned_history[:, 2], label='V_down (m/s)')
        axs[1].set_ylabel('Velocity (NED) [m/s]')
        axs[1].legend()
        axs[1].grid(True)

        # Plot Euler Angles
        axs[2].plot(time_history, euler_angles_history[:, 0], label='Roll (deg)')
        axs[2].plot(time_history, euler_angles_history[:, 1], label='Pitch (deg)')
        axs[2].plot(time_history, euler_angles_history[:, 2], label='Yaw (deg)')
        axs[2].set_ylabel('Euler Angles [deg]')
        axs[2].legend()
        axs[2].grid(True)

        # Plot Angular Velocity (Body Frame)
        axs[3].plot(time_history, rot_body_history[:, 0], label='p (roll rate, deg/s)')
        axs[3].plot(time_history, rot_body_history[:, 1], label='q (pitch rate, deg/s)')
        axs[3].plot(time_history, rot_body_history[:, 2], label='r (yaw rate, deg/s)')
        axs[3].set_ylabel('Angular Velocity (Body) [deg/s]')
        axs[3].legend()
        axs[3].grid(True)

        # Plot Control Actions from RL Agent
        # Note: time_history has one more element than control_action_history (initial state vs actions during steps)
        axs[4].plot(time_history[1:], control_action_history[:, 0], label='Rotor Speed 1 (rad/s)')
        axs[4].plot(time_history[1:], control_action_history[:, 1], label='Rotor Speed 2 (rad/s)')
        axs[4].plot(time_history[1:], control_action_history[:, 2], label='Elevon Deflection 1 (rad)')
        axs[4].plot(time_history[1:], control_action_history[:, 3], label='Elevon Deflection 2 (rad)')
        axs[4].set_ylabel('Control Actions')
        axs[4].set_xlabel('Time (s)')
        axs[4].legend()
        axs[4].grid(True)


        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        plt.show()


    except FileNotFoundError as e:
        print(f"ERROR: A required file was not found.")
        print(f"Details: {e}")
        print("Please ensure 'mavion_model.py', 'rotation.py', and 'mavion.yaml' are all present in the same directory as this script.")
    except ImportError as e:
        print(f"ERROR: Could not import a required module.")
        if 'matplotlib' in str(e).lower():
            print("Details: Matplotlib library not found. Please install it (e.g., 'pip install matplotlib numpy').")
        elif 'mavion_model' in str(e).lower() or 'rotation' in str(e).lower():
             print(f"Details: {e}. Could not import 'mavion_model' or 'rotation'.")
             print("Please ensure 'mavion_model.py' and 'rotation.py' are in the same directory or accessible in your Python path.")
        else:
            print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_mavion_simulation_timesteps_with_plotting()
