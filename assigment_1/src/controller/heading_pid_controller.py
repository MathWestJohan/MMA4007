from controller.base_controller import BaseController
import numpy as np


class PIDController(BaseController):
    def __init__(self):
        # Heading (rudder) gains
        self.kp_psi = 0.3
        self.ki_psi = 0.0
        self.kd_psi = 3.5

        # Anti-windup limit
        self.psi_int_limit = np.deg2rad(20)

        self.reset()

    def reset(self):
        self.e_psi_int = 0.0
        self.e_psi_prev = 0.0

    def get_command(self, state, target, dt):
        """
        target.psi -> desired heading (rad)
        """

        # Fixed propulsion
        rpm = 80.0

        # Heading error (wrapped to [-pi, pi])
        e_psi = (target.psi - state.psi + np.pi) % (2*np.pi) - np.pi

        # Integral term with anti-windup
        self.e_psi_int += e_psi * dt
        self.e_psi_int = np.clip(self.e_psi_int,
                                 -self.psi_int_limit,
                                  self.psi_int_limit)

        # Derivative term
        de_psi = (e_psi - self.e_psi_prev) / dt

        # PD control
        rudder = (
            self.kp_psi * e_psi +
            self.ki_psi * self.e_psi_int +
            self.kd_psi * de_psi
        )

        self.e_psi_prev = e_psi

        return rpm, np.rad2deg(rudder)
