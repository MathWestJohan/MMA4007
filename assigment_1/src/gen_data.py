from sim_env.simulator import Sim, ShipState
from controller.heading_pid_controller import PIDController
import numpy as np
import pandas as pd
import os


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def gen_data(save_path):
    num = 200  # traj num
    psi_arr = np.random.random(num)*(2*np.pi)  # random initial psi range from [0, 2*pi]
    target_psi_arr = np.random.random(num) * (2 * np.pi) # random target heading psi range from [0, 2*pi]

    os.makedirs(save_path, exist_ok=True)  # Create folder if it does not exist

    for i in range(num):
        state0 = ShipState(500, 400, psi_arr[i], 0, 0, 0)
        
        target_psi = target_psi_arr[i]
        target = ShipState(0,0, target_psi, 10, 0, 0)
        
        controller = PIDController()
        sim = Sim(state0, target, controller, dt = 1.0, tf = 200)
        history = sim.run()
        
        df = pd.DataFrame(history)
        
        # Store target + heading error for training
        df["target_psi"] = target_psi
        df["e_psi"] = wrap_to_pi(target_psi - df["psi"].values)
        
        file_name = os.path.join(save_path, f"demonstration_data{i+1}.csv")
        df.to_csv(file_name, index=False)


if __name__ == '__main__':
    save_path = os.path.join(os.path.dirname(__file__), "demonstration_data")
    gen_data(save_path)
