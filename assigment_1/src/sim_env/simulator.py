import numpy as np
from dataclasses import dataclass
from controller.base_controller import BaseController
from sim_env.ship_model import ShipModel
from sim_env.actuator import Actuator


@dataclass
class ShipState:
    x: float
    y: float
    psi: float  # yaw
    u: float
    v: float
    r: float

class Sim:
    def __init__(self, state0: ShipState, target: ShipState, controller: BaseController, dt, tf):
        """
        Generic simulator.

        Parameters
        ----------
        state0 : initial state
        target : target state
        controller: 
            Controller object with `get_command(state, target, dt)` method.
        dt : float
            Simulation timestep.
        tf : float
            Final simulation time.
        """
        self.state0 = state0
        self.target = target
        self.controller = controller
        self.model = ShipModel()
        self.actuator = Actuator()
        self.dt = dt
        self.T = np.arange(0, tf, dt)

    def state_to_array(self, state: ShipState):
        return np.array([state.x, state.y, state.psi, state.u, state.v, state.r], dtype=float)

    def array_to_state(self, arr):
        return ShipState(*arr)

    def run(self):
        """
        Run the docking simulation.

        Returns
        -------
        success : bool
            True if ship docks within threshold.
        history : list of dict
            Time series of state, errors, PID commands, and actuator outputs.
        """
        state = self.state0
        history = []
        self.controller.reset()

        for t in self.T:

            # Get controller commands
            cmd_rpm, cmd_rudder = self.controller.get_command(state, self.target, self.dt)

            # Actuator output
            act = self.actuator.step(cmd_rpm, cmd_rudder, self.dt)

            # RK4 integration
            def f(s):
                return self.model.dynamics(s, act)

            x = self.state_to_array(state)
            k1 = f(state)
            k2 = f(self.array_to_state(x + 0.5*self.dt*k1))
            k3 = f(self.array_to_state(x + 0.5*self.dt*k2))
            k4 = f(self.array_to_state(x + self.dt*k3))
            x += self.dt*(k1 + 2*k2 + 2*k3 + k4)/6
            state = self.array_to_state(x)
            state.psi = state.psi % (2 * np.pi)

            # Log data for training
            history.append({
                't': t,
                'x': state.x,
                'y': state.y,
                'psi': state.psi,
                'u': state.u,
                'v': state.v,
                'r': state.r,
                'cmd_rpm': cmd_rpm,
                'cmd_rudder': cmd_rudder,
                'act_rpm': act[0],
                'act_rudder': act[1]
            })

        return history