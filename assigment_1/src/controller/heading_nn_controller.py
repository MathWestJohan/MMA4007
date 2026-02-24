import torch
import torch.nn as nn
import numpy as np
from controller.base_controller import BaseController


# Same model definition used during training
class RudderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class RudderNNController(BaseController):
    def __init__(self, model_path, device="cpu", default_rpm=0.0):
        self.device = device
        self.default_rpm = default_rpm

        self.model = RudderModel().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=self.device)
        )
        self.model.eval()

    def reset(self):
        # No internal state (stateless controller)
        pass

    def get_command(self, state, target, dt: float):
        """
        Args:
            state: object or dict containing 'psi', 'u', 'v', 'r'
            target: unused (kept for interface compatibility)
            dt: timestep

        Returns:
            cmd_rpm (float)
            cmd_rudder (deg)
        """

        # --- extract state ---
        psi = state.psi
        u = state.u
        v = state.v
        r = state.r
        
        e_psi = (target.psi - psi + np.pi) % (2 * np.pi) - np.pi # heading error, wrapped to [-pi, pi]
        
        x_tensor = torch.tensor([[e_psi, u, v, r]], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            cmd_rudder = self.model(x_tensor).item()
            
        cmd_rudder = float(np.clip(cmd_rudder, -45.0, 45.0))
        cmd_rpm = self.default_rpm
        return cmd_rpm, cmd_rudder