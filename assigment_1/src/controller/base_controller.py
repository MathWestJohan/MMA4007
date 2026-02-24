import numpy as np


class BaseController:
    def reset(self):
        pass

    def get_command(self, state, target, dt: float):
        """
        Returns:
            cmd_rpm (float)
            cmd_rudder (deg)
        """
        raise NotImplementedError