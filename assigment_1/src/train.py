import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class RudderDataset(Dataset):
    def __init__(self, folder_path):
        self.data = []

        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(folder_path, file))

                # Take all rows (skip the first if needed)
                e_psi = df["e_psi"].values[1:] # use heading error as input
                u = df["u"].values[1:]
                v = df["v"].values[1:]
                r = df["r"].values[1:]
                cmd_rudder = df["cmd_rudder"].values[1:]

                # Collect as (input_vector, output)
                for e, uu, vv, rr, c in zip(e_psi, u, v, r, cmd_rudder):
                    self.data.append(([e, uu, vv, rr], c))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return (
            torch.tensor(x, dtype=torch.float32),  # shape [4]
            torch.tensor([y], dtype=torch.float32) # shape [1]
        )


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


# Paths
data_folder = os.path.join(os.path.dirname(__file__), "demonstration_data")

# Dataset & DataLoader
dataset = RudderDataset(data_folder)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

# Model, loss, optimizer
model = RudderModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# Training
epochs = 100
for epoch in range(epochs):
    total_loss = 0.0
    for psi, cmd_rudder in loader:
        pred = model(psi)
        loss = criterion(pred, cmd_rudder)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

# save the model
model_path = os.path.join(os.path.dirname(__file__), "rudder_model.pt")
torch.save(model.state_dict(), model_path)
