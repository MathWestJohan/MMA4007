import numpy as np


class Actuator:
    def __init__(self):
        self.rpm = 80.0
        self.rudder = 0.0

        self.rpm_max = 132
        self.rpm_rate = 13

        self.rudder_max = 45
        self.rudder_rate = 3.7

    def step(self, cmd_rpm, cmd_rudder, dt):
        self.rpm += np.clip(cmd_rpm - self.rpm,
                            -self.rpm_rate*dt,
                             self.rpm_rate*dt)
        self.rpm = np.clip(self.rpm, -self.rpm_max, self.rpm_max)

        self.rudder += np.clip(cmd_rudder - self.rudder,
                               -self.rudder_rate*dt,
                                self.rudder_rate*dt)
        self.rudder = np.clip(self.rudder,
                               -self.rudder_max,
                                self.rudder_max)

        return np.array([self.rpm, self.rudder])
    
