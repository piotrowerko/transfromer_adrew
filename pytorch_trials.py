import torch
from torch.utils.data import DataLoader, TensorDataset

# Define some sample data
X = torch.randn(1000, 10)  # input features
y = torch.randint(0, 2, (1000, 1))  # binary labels

# Create a TensorDataset object from the data
dataset = TensorDataset(X, y)

# Create a DataLoader object with batch size 32
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over the dataloader and print out some batches
for i, (batch_x, batch_y) in enumerate(dataloader):
    print(f"Batch {i}: input shape {batch_x.shape}, label shape {batch_y.shape}")