import numpy as np
import pandas as pd
from pathlib import Path


class ShipModel:
    def __init__(self, rpm_file=None, rudder_file=None):

        # Get the directory of this script
        base_path = Path(__file__).parent

        # Use default paths if none provided
        if rpm_file is None:
            rpm_file = base_path / "forces_data" / "rpm.txt"
        if rudder_file is None:
            rudder_file = base_path / "forces_data" / "rudder.txt"

        self.M = np.array([
            [1.1e7, 0, 0],
            [0, 1.1e7, 8.4e6],
            [0, 8.4e6, 5.8e9]
        ])
        self.D = np.array([
            [3.0e5, 0, 0],
            [0, 5.5e5, 6.4e5],
            [0, 6.4e5, 1.2e8]
        ])
        self.M_inv = np.linalg.inv(self.M)

        # Load propulsion data
        Dt = pd.read_csv(rpm_file, names=["rpm", "coef"])
        self.rpm_poly = np.polyfit(Dt.rpm, Dt.coef, 5)

        Dr = pd.read_csv(rudder_file,
                         names=["angle_lift", "lift_coef",
                                "angle_prop", "prop_coef"])
        self.lift_poly = np.polyfit(Dr.angle_lift, Dr.lift_coef, 5)
        self.prop_poly = np.polyfit(Dr.angle_prop, Dr.prop_coef, 5)

    def forces(self, rpm, rudder_deg):
        arm = -41.5
        base = np.polyval(self.rpm_poly, rpm) * 1e6

        prop = np.polyval(self.prop_poly, rudder_deg) * base
        lift = np.polyval(self.lift_poly, rudder_deg) * base

        if rpm > 0:
            return np.array([2*prop, 2*lift, 2*arm*lift])
        elif rpm < 0:
            return np.array([2*base, 0, 0])
        else:
            return np.zeros(3)

    def dynamics(self, state, act):
        F = self.forces(act[0], act[1])

        nu = np.array([state.u, state.v, state.r])
        nu_dot = self.M_inv @ (F - self.D @ nu)

        R = np.array([
            [np.cos(state.psi), -np.sin(state.psi), 0],
            [np.sin(state.psi),  np.cos(state.psi), 0],
            [0, 0, 1]
        ])

        eta_dot = R @ nu

        return np.hstack((eta_dot, nu_dot))