import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import pandas as pd

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Load dataset
loader = EnglandCovidDatasetLoader()
dataset = loader.get_dataset(lags=8)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# Get dataset information
snapshot = next(iter(train_dataset))
node_count = snapshot.x.shape[0]
feature_count = snapshot.x.shape[1]
print(f"Node count: {node_count}, Feature count: {feature_count}")

# Define MPNN-LSTM model
class MPNN_LSTM(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout):
        super(MPNN_LSTM, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        self.nhid = nhid
        self.nfeat = nfeat
        
        # Graph convolution layers
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.rnn1 = nn.LSTM(2*nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)
        
        self.fc1 = nn.Linear(2*nhid+nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, edge_attr=None):
        batch_size = 1
        lst = []
        
        # Create a skip connection
        skip = x.clone()
        
        # First graph convolutional layer
        x = self.relu(self.conv1(x, edge_index, edge_weight=edge_attr))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        
        # Second graph convolutional layer
        x = self.relu(self.conv2(x, edge_index, edge_weight=edge_attr))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        
        # Concatenate features from both GCN layers
        x = torch.cat(lst, dim=1)
        
        # Reshape for LSTM
        x = x.unsqueeze(0)
        
        # LSTM layers
        x, (hn1, cn1) = self.rnn1(x)
        out2, (hn2, cn2) = self.rnn2(x)
        
        # Concatenate hidden states
        x = torch.cat([hn1[0], hn2[0]], dim=1)
        
        # Skip connection
        skip_avg = torch.mean(skip, dim=0, keepdim=True)
        skip_avg = skip_avg.expand(x.size(0), -1)
        
        # Combine with skip connection
        x = torch.cat([x, skip_avg], dim=1)
        
        # Final fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x).squeeze()
        
        return x

# Prepare data for CatBoost
def prepare_data_for_catboost(dataset):
    X = []
    y = []
    node_indices = []
    timestamps = []
    
    timestamp = 0
    for snapshot in dataset:
        features = snapshot.x.numpy()
        targets = snapshot.y.numpy()
        
        for i in range(node_count):
            X.append(features[i])
            y.append(targets[i])
            node_indices.append(i)
            timestamps.append(timestamp)
        
        timestamp += 1
    
    X = np.array(X)
    y = np.array(y)
    
    # Create a DataFrame for easier handling
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['node_index'] = node_indices
    df['timestamp'] = timestamps
    df['target'] = y
    
    return df

# Prepare data
train_df = prepare_data_for_catboost(train_dataset)
test_df = prepare_data_for_catboost(test_dataset)

# Train CatBoost model
print("Training CatBoost model...")
catboost_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    loss_function='RMSE',
    verbose=100,
    random_seed=seed
)

# Extract features and target
X_train = train_df.drop(['target', 'node_index', 'timestamp'], axis=1)
y_train = train_df['target']
X_test = test_df.drop(['target', 'node_index', 'timestamp'], axis=1)
y_test = test_df['target']

# Train CatBoost
catboost_model.fit(X_train, y_train)

# CatBoost predictions
catboost_predictions = catboost_model.predict(X_test)
catboost_mse = mean_squared_error(y_test, catboost_predictions)
catboost_mae = mean_absolute_error(y_test, catboost_predictions)
catboost_r2 = r2_score(y_test, catboost_predictions)

print(f"CatBoost Test MSE: {catboost_mse:.4f}")
print(f"CatBoost Test MAE: {catboost_mae:.4f}")
print(f"CatBoost Test R²: {catboost_r2:.4f}")

# Train MPNN-LSTM model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mpnn_model = MPNN_LSTM(
    nfeat=feature_count,
    nhid=32,  # Increased from 16
    nout=1,
    n_nodes=node_count,
    window=8,
    dropout=0.2
).to(device)

optimizer = torch.optim.Adam(mpnn_model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
criterion = nn.MSELoss()

# Training loop with early stopping
best_val_loss = float('inf')
patience = 20
counter = 0
epochs = 200

mpnn_model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for snapshot in train_dataset:
        optimizer.zero_grad()
        
        x = snapshot.x.to(device)
        edge_index = snapshot.edge_index.to(device)
        edge_attr = snapshot.edge_attr.to(device) if hasattr(snapshot, 'edge_attr') else None
        y = snapshot.y.to(device)
        
        y_hat = mpnn_model(x, edge_index, edge_attr)
        loss = criterion(y_hat, y)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Validation
    if epoch % 5 == 0:
        mpnn_model.eval()
        val_loss = 0
        with torch.no_grad():
            for snapshot in test_dataset:
                x = snapshot.x.to(device)
                edge_index = snapshot.edge_index.to(device)
                edge_attr = snapshot.edge_attr.to(device) if hasattr(snapshot, 'edge_attr') else None
                y = snapshot.y.to(device)
                
                y_hat = mpnn_model(x, edge_index, edge_attr)
                val_loss += criterion(y_hat, y).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save the best model
            torch.save(mpnn_model.state_dict(), "best_mpnn_model.pth")
        else:
            counter += 1
        
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        mpnn_model.train()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}: Loss {epoch_loss:.4f}")

# Load best model
mpnn_model.load_state_dict(torch.load("best_mpnn_model.pth"))

# Evaluation
mpnn_model.eval()
mpnn_predictions = []
mpnn_targets = []

with torch.no_grad():
    for snapshot in test_dataset:
        x = snapshot.x.to(device)
        edge_index = snapshot.edge_index.to(device)
        edge_attr = snapshot.edge_attr.to(device) if hasattr(snapshot, 'edge_attr') else None
        y = snapshot.y.to(device)
        
        y_hat = mpnn_model(x, edge_index, edge_attr)
        
        mpnn_predictions.extend(y_hat.cpu().numpy())
        mpnn_targets.extend(y.cpu().numpy())

# Calculate metrics
mpnn_mse = mean_squared_error(mpnn_targets, mpnn_predictions)
mpnn_mae = mean_absolute_error(mpnn_targets, mpnn_predictions)
mpnn_r2 = r2_score(mpnn_targets, mpnn_predictions)

print(f"MPNN-LSTM Test MSE: {mpnn_mse:.4f}")
print(f"MPNN-LSTM Test MAE: {mpnn_mae:.4f}")
print(f"MPNN-LSTM Test R²: {mpnn_r2:.4f}")

# Plot results
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(y_test, label="True Values", color='blue', alpha=0.7)
plt.plot(catboost_predictions, label="CatBoost Predictions", color='green', linestyle='--')
plt.legend()
plt.title("CatBoost Predictions")
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(mpnn_targets, label="True Values", color='blue', alpha=0.7)
plt.plot(mpnn_predictions, label="MPNN-LSTM Predictions", color='red', linestyle='--')
plt.legend()
plt.title("MPNN-LSTM Predictions")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()

# Save results for dashboard
results = {
    'true_values': y_test,
    'catboost_predictions': catboost_predictions,
    'mpnn_predictions': mpnn_predictions,
    'catboost_metrics': {
        'mse': catboost_mse,
        'mae': catboost_mae,
        'r2': catboost_r2
    },
    'mpnn_metrics': {
        'mse': mpnn_mse,
        'mae': mpnn_mae,
        'r2': mpnn_r2
    }
}

# Save results
np.savez('model_results.npz', 
         true_values=y_test, 
         catboost_predictions=catboost_predictions, 
         mpnn_predictions=mpnn_predictions,
         node_indices=test_df['node_index'],
         timestamps=test_df['timestamp'])

print("Results saved for dashboard.")