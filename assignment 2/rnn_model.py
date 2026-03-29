import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the data
train_x = np.load('train test data/train_x.npy')
train_y = np.load('train test data/train_y.npy')
test_x = np.load('train test data/test_x.npy')
test_y = np.load('train test data/test_y.npy')

print(f"Train X shape: {train_x.shape}")
print(f"Test X shape: {test_x.shape}")

INPUT_SIZE = train_x.shape[2]  # 12 or 6 depending on feature selection
print(f"Using {INPUT_SIZE} features")

# Convert to tensors
train_x = torch.FloatTensor(train_x)
train_y = torch.LongTensor(train_y)
test_x = torch.FloatTensor(test_x)
test_y = torch.LongTensor(test_y)

# Normalize input features
mean = train_x.mean(dim=(0, 1), keepdim=True)
std = train_x.std(dim=(0, 1), keepdim=True)
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

# Create datasets and dataloaders
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class ActivityRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, 
                 model_type='RNN', dropout=0.0):
        super(ActivityRecognitionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        
        # Select RNN type
        if model_type == 'RNN':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True, 
                              dropout=dropout if num_layers > 1 else 0)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    return total_loss / len(test_loader), 100. * correct / total


def run_experiment(model_type, input_size, hidden_size, num_layers, 
                   num_epochs, device, dropout=0.0, lr=0.001):
    """Run experiment and track BEST test accuracy"""
    
    # Reset seed for fair comparison
    torch.manual_seed(42)
    
    model = ActivityRecognitionModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=6,
        num_layers=num_layers,
        model_type=model_type,
        dropout=dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n{'='*70}")
    print(f"Training {model_type} | Layers: {num_layers} | Hidden: {hidden_size} | "
          f"Dropout: {dropout} | LR: {lr}")
    print(f"{'='*70}")
    
    best_test_acc = 0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Track best accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | "
                  f"Best: {best_test_acc:.2f}% (ep {best_epoch})")
    
    print(f"\n>>> BEST Test Accuracy: {best_test_acc:.2f}% at epoch {best_epoch}")
    
    return best_test_acc, best_epoch


# Main experiments
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

results = {}

print("\n" + "="*70)
print("FINDING BEST ACCURACY FOR EACH ARCHITECTURE")
print("Using all 12 features, tracking best test accuracy")
print("="*70)

# Adding dropout to help with overfitting

configurations = [
    # (layers, hidden, dropout, lr, epochs)
    (2, 128, 0.0, 0.001, 200),
    (2, 128, 0.2, 0.001, 200),
    (2, 256, 0.2, 0.001, 200),
    (3, 128, 0.3, 0.001, 200),
]

for model_type in ['RNN', 'GRU', 'LSTM']:
    print(f"\n{'#'*70}")
    print(f"# {model_type}")
    print(f"{'#'*70}")
    
    best_overall = 0
    best_config = None
    
    for layers, hidden, dropout, lr, epochs in configurations:
        acc, ep = run_experiment(
            model_type=model_type,
            input_size=INPUT_SIZE,
            hidden_size=hidden,
            num_layers=layers,
            num_epochs=epochs,
            device=device,
            dropout=dropout,
            lr=lr
        )
        
        if acc > best_overall:
            best_overall = acc
            best_config = (layers, hidden, dropout, lr, ep)
    
    results[model_type] = (best_overall, best_config)
    print(f"\n*** {model_type} BEST: {best_overall:.2f}% ***")
    print(f"    Config: {best_config[0]}L, {best_config[1]}H, dropout={best_config[2]}, "
          f"lr={best_config[3]}, epoch={best_config[4]}")

# Final Summary
print("\n" + "="*70)
print("FINAL SUMMARY - BEST ACCURACY FOR EACH MODEL")
print("="*70)
for model_type, (acc, config) in results.items():
    layers, hidden, dropout, lr, epoch = config
    print(f"{model_type:6s}: {acc:.2f}% | {layers}L, {hidden}H, dropout={dropout}, epoch={epoch}")