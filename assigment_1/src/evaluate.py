from controller.heading_nn_controller import RudderNNController
from sim_env.simulator import Sim, ShipState
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wrap_to_pi(angle):
  return (angle + np.pi) % (2 * np.pi) - np.pi

def run_case(init_psi_deg, target_psi_deg, model_path="rudder_model.pt"):
  """
  Simulates a ship maneuvering scenario using a neural network-based rudder controller and evaluates its performance.
  Args:
    init_psi_deg (float): Initial heading angle of the ship in degrees.
    target_psi_deg (float): Target heading angle of the ship in degrees.
    model_path (str, optional): Path to the trained neural network model for the rudder controller. Defaults to "rudder_model.pt".
  Returns:
    tuple:
      - df (pd.DataFrame): DataFrame containing the simulation results for each timestep.
      - final_err_deg (float): Final heading error in degrees at the end of the simulation.
      - mae_deg (float): Mean absolute heading error in degrees over the entire simulation.
  """
  state0 = ShipState(500, 400, np.deg2rad(init_psi_deg), 0, 0, 0)
  target = ShipState(0, 0, np.deg2rad(target_psi_deg), 10, 0, 0)
  
  nn = RudderNNController(model_path = model_path, default_rpm = 80.0)
  sim = Sim(state0, target, nn, dt = 1.0, tf = 200)
  df = pd.DataFrame(sim.run())
  
  e_psi = wrap_to_pi(target.psi - df["psi"].values)
  final_err_deg = np.rad2deg(e_psi[-1])
  mae_deg = np.rad2deg(np.mean(np.abs(e_psi)))
  
  return df, final_err_deg, mae_deg

eval_cases = [
  (180, 40),
  (0, 90),
  (270, 45),
  (30, 210),
  (350, 10),
  (120, 300),
]

plt.figure()
rows = []

for init_deg, tgt_deg in eval_cases:
  df, final_err_deg, mae_deg = run_case(init_deg, tgt_deg)
  plt.plot(df["x"], df["y"], label=f"{init_deg}->{tgt_deg}")
  rows.append({
    "init_psi_deg": init_deg,
    "target_psi_deg": tgt_deg,
    "final_heading_error_deg": final_err_deg,
    "mean_abs_heading_error_deg": mae_deg,
  })
  
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("NN trajectories across multiple heading targets")
plt.axis("equal")
plt.show()

print(pd.DataFrame(rows))
