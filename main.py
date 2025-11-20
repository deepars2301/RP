import numpy as np
from stable_baselines3 import TD3, SAC  # Still imported if you need it later
import matplotlib.pyplot as plt
from mavion_model import Mavion
from mavion_env import MavionEnv
from rotation import *

def upulse(t, t0, tf):
    return 0 if (t < t0 or t >= tf) else 1

def ustep(t, t0):
    return 0 if t < t0 else 1

def uramp(t, t0):
    return 0 if t < t0 else (t - t0)

if __name__ == "__main__":
    
    # Get desired hover altitude from the user
    desired_altitude = float(input("Enter desired hover altitude (in meters): "))
    
    mav_env = MavionEnv()

    # For hovering, set both horizontal and vertical velocities to zero.
    vh = 0
    vz = 0  
    cap = 0 / rad2deg
    omega = 0

    # Use hover target (zero velocities)
    target = [vh, vz]
    
    # Compute trim values for hover
    dx, de, theta = mav_env.mavion.trim(target)
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    print("Trim values:", dx, de, theta * rad2deg)
    
    # Define the initial state (starting at ground level)
    pos = np.array([0, 0, -5])
    ang = np.array([0, 90*np.pi/180, 0]) 
    quat = eul2quat(ang)
    vel = np.zeros(3)
    rot = np.zeros(3)
    
    x0 = np.concatenate([pos, quat, vel, rot])
    
    # Set the target state so that the drone hovers at the user-specified altitude.
    # Note: the simulation’s coordinate system uses negative z for altitude.
    pos_target = np.array([0, 0, -5])
    eul_target = np.array([0, 90*np.pi/180, 0])  # Level flight with a heading of 90°
    quat_target = eul2quat(eul_target)
    vel_target = np.zeros(3)
    rot_target = np.zeros(3)
    mav_env.target = np.concatenate([pos_target, quat_target, vel_target, rot_target])
    
    # >>> Option 1: Do not load the RL agent for hover control <<<
    # Comment out or remove the following agent loading code:
    mav_agent = SAC.load("TrainingData_TD3\SAC_hover_Total4MSteps.zip", env=mav_env)
    mav_env.reset()
    x0 = mav_env.state

    # Use constant control inputs (trim values) for hover.
    def ctl(t, s):
        act, _ = mav_agent.predict(s, deterministic=True)
        u = act * mav_env.action_range
        #target = [0, 0]
        #dx, de, theta= mav_env.mavion.trim(target)
        #u= [dx, dx, de, de]
        return u
    
    def wnd(t, s):
        return np.zeros(3)
    
    print("Initial state:", x0, "Control input:", ctl(0, x0))
    
    # Run the simulation for 120 seconds.
    sim = mav_env.mavion.sim((0,10),x0, ctl, wnd)
    
    # Plot the results.
    fig, axs = plt.subplots(2, 2)
    
    axs[0][0].plot(sim.t, sim.y[0:3].T, label=('x', 'y', 'z'))
    axs[0][0].legend()
    axs[0][0].grid()
    axs[0][0].set_title('Positions')
    
    axs[1][0].plot(sim.t, sim.y[3:7].T, label=('q0', 'q1', 'q2', 'q3'))
    axs[1][0].legend()
    axs[1][0].grid()
    axs[1][0].set_title('Quaternions')
    
    axs[0][1].plot(sim.t, sim.y[7:10].T, label=('Vn', 'Ve', 'Vz'))
    axs[0][1].legend()
    axs[0][1].grid()
    axs[0][1].set_title('Velocities')
    
    axs[1][1].plot(sim.t, sim.y[10:13].T, label=('p', 'q', 'r'))
    axs[1][1].legend()
    axs[1][1].grid()
    axs[1][1].set_title('Rotational Rates')
    
    plt.show()
