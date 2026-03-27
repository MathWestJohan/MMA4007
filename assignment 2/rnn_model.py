import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Load the data
train_x = np.load('train test data/train_x.npy')
train_y = np.load('train test data/train_y.npy')
test_x = np.load('train test data/test_x.npy')
test_y = np.load('train test data/test_y.npy')

print(f"Train X shape: {train_x.shape}")  # (2205, 50, 12)
print(f"Train Y shape: {train_y.shape}")  # (2205,)
print(f"Test X shape: {test_x.shape}")    # (430, 50, 12)
print(f"Test Y shape: {test_y.shape}")    # (430,)

# Keeping only the last 6 features (Rotationrate and acceleration)
train_x = train_x[:, :, 6:]
test_x = test_x[:, :, 6:]

print(f"\nAfter feature selection:")
print(f"Train X shape: {train_x.shape}")
print(f"Test X shape: {test_x.shape}")

# Conver to tensors and create datasets
train_x = torch.FloatTensor(train_x)
train_y = torch.LongTensor(train_y)
test_x = torch.FloatTensor(test_x)
test_y = torch.LongTensor(test_y)

# Normalize data x and y
mean = train_x.mean(axis=(0, 1), keepdim=True)
std = train_x.std(axis=(0, 1))
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

# Create datasets and dataloaders
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Human activity recognition model using RNN
# Start by using the last 6 features (Rotationrate and acceleration)
# and vanilla RNN with a single layer. What is the accuracy of the model?

class ActivityRecognitionRNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers=1, model_type='RNN'):
    super(ActivityRecognitionRNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.model_type = model_type
    
    # Select RNN type
    if model_type == 'RNN':
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True)
    elif model_type == 'GRU':
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True)
    elif model_type == 'LSTM':
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    self.fc = nn.Linear(hidden_size, output_size)  # Assuming 6 activity classes
    
  def forward(self, x):
      """ 
      x: input of shape (batch_size, seq_length, input_size)
      """
      rnn_out, _ = self.rnn(x)  # Get the output of the RNN
      last_time_step = rnn_out[:, -1, :]  # Get the output of the last time step
      output = self.fc(last_time_step)  # Pass through the fully connected layer
      
      return output

# Training model  
def train_model(model, train_loader, criterion, optimizer, device):
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

# Evaluating model
def evaluate_model(model, test_loader, criterion, device):
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

def run_experiment(model_type, input_size, hidden_size, num_layers, num_epochs, device):
    """Run a single experiment and return final test accuracy"""
    
    model = ActivityRecognitionRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=6,
        num_layers=num_layers,
        model_type=model_type
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n{'='*60}")
    print(f"Training {model_type} | Layers: {num_layers} | Hidden: {hidden_size} | Features: {input_size}")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    final_test_loss, final_test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    
    return final_test_acc
 
 
# Main experiments
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
 
results = {}
 
# Experiment 1: Compare RNN vs GRU vs LSTM (1 layer, 128 hidden, 6 features)
print("\n" + "="*70)
print("EXPERIMENT: Comparing RNN, GRU, and LSTM (1 layer, 128 hidden units)")
print("="*70)
 
for model_type in ['RNN', 'GRU', 'LSTM']:
    # Reset seed for fair comparison
    torch.manual_seed(42)
    acc = run_experiment(model_type, input_size=6, hidden_size=128, 
                         num_layers=2, num_epochs=100, device=device)
    results[f"{model_type}_1layer_128hidden"] = acc
 
# Print summary
print("\n" + "="*70)
print("SUMMARY OF RESULTS")
print("="*70)
for name, acc in results.items():
    print(f"{name}: {acc:.2f}%")