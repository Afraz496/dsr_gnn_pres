import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import GMAN
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the chickenpox dataset
loader = ChickenpoxDatasetLoader()
dataset = loader.get_dataset()

# Split the dataset into train and test
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# Process the data for both models
# For GMAN: prepare graph data
edge_indices = []
edge_weights = []
features = []
targets = []

for snapshot in train_dataset:
    edge_indices.append(snapshot.edge_index)
    edge_weights.append(snapshot.edge_attr)
    features.append(snapshot.x)
    targets.append(snapshot.y)

test_edge_indices = []
test_edge_weights = []
test_features = []
test_targets = []

for snapshot in test_dataset:
    test_edge_indices.append(snapshot.edge_index)
    test_edge_weights.append(snapshot.edge_attr)
    test_features.append(snapshot.x)
    test_targets.append(snapshot.y)

# Convert to PyTorch tensors
features = torch.stack(features)
targets = torch.stack(targets)
test_features = torch.stack(test_features)
test_targets = torch.stack(test_targets)

# Prepare data for CatBoost
def prepare_catboost_data(features, targets):
    # Flatten features and add node indices
    n_nodes = features.shape[1]
    n_features = features.shape[2]
    n_time_steps = features.shape[0]
    
    # Create a DataFrame with time, node, and feature information
    catboost_data = []
    
    for t in range(n_time_steps):
        for node in range(n_nodes):
            row = {
                'time_step': t,
                'node_id': node
            }
            
            # Add features
            for f in range(n_features):
                row[f'feature_{f}'] = features[t, node, f].item()
            
            # Add target
            row['target'] = targets[t, node].item()
            
            catboost_data.append(row)
    
    return pd.DataFrame(catboost_data)

# Prepare train and test data for CatBoost
train_df = prepare_catboost_data(features, targets)
test_df = prepare_catboost_data(test_features, test_targets)

# Define the GMAN model
class ChickenpoxGMAN(nn.Module):
    def __init__(self, node_features, num_nodes, time_steps=12):
        super(ChickenpoxGMAN, self).__init__()
        
        # Get dimensions of data
        self.num_nodes = num_nodes
        self.node_features = node_features
        
        # Create a custom implementation inspired by GMAN since the original has different parameters
        self.spatial_attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.temporal_attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        
        # Input and output projections
        self.input_projection = nn.Linear(node_features, 64)
        self.output_projection = nn.Linear(64, 1)
        
        # Normalization layers
        self.spatial_norm = nn.LayerNorm(64)
        self.temporal_norm = nn.LayerNorm(64)
        
    def forward(self, x, edge_index, edge_weight):
        # x shape: [batch_size, num_nodes, node_features]
        batch_size, num_nodes, _ = x.shape
        
        # Project input features to embedding dimension
        x = self.input_projection(x)  # [batch_size, num_nodes, 64]
        
        # Spatial attention across nodes
        x_reshaped = x.view(batch_size, num_nodes, 64)
        x_perm = x_reshaped.permute(1, 0, 2)  # [num_nodes, batch_size, 64]
        
        # Apply spatial attention
        spatial_out, _ = self.spatial_attention(x_perm, x_perm, x_perm)
        spatial_out = spatial_out.permute(1, 0, 2)  # [batch_size, num_nodes, 64]
        
        # Residual connection and normalization
        x = self.spatial_norm(x + spatial_out)
        
        # Temporal attention - if we had multiple time steps in batch
        # For chickenpox data, we'll just use the spatial attention
        
        # Output projection
        out = self.output_projection(x)  # [batch_size, num_nodes, 1]
        return out.squeeze(-1)  # [batch_size, num_nodes]

# Define training parameters
num_nodes = features.shape[1]
node_features = features.shape[2]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
gman_model = ChickenpoxGMAN(node_features, num_nodes).to(device)
catboost_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=100
)

# Define optimizer for GMAN
optimizer = torch.optim.Adam(gman_model.parameters(), lr=0.001)

# Train GMAN model
def train_gman(model, optimizer, features, targets, edge_indices, edge_weights, epochs=100):
    model.train()
    training_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = []
        for t in range(len(features)):
            x = features[t].to(device)
            edge_index = edge_indices[t].to(device)
            edge_weight = edge_weights[t].to(device)
            
            y_hat = model(x.unsqueeze(0), edge_index, edge_weight)
            predictions.append(y_hat.squeeze(0))
        
        predictions = torch.stack(predictions)
        loss = F.mse_loss(predictions, targets.to(device))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        training_losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return training_losses

# Train CatBoost model
def train_catboost(model, train_df):
    X = train_df.drop('target', axis=1)
    y = train_df['target']
    
    model.fit(X, y)
    return model

# Evaluate models
def evaluate_models(gman_model, catboost_model, test_features, test_targets, test_edge_indices, test_edge_weights, test_df):
    # Evaluate GMAN
    gman_model.eval()
    gman_predictions = []
    
    with torch.no_grad():
        for t in range(len(test_features)):
            x = test_features[t].to(device)
            edge_index = test_edge_indices[t].to(device)
            edge_weight = test_edge_weights[t].to(device)
            
            y_hat = gman_model(x.unsqueeze(0), edge_index, edge_weight)
            gman_predictions.append(y_hat.squeeze(0))
    
    gman_predictions = torch.cat(gman_predictions).cpu().numpy()
    test_targets_flat = test_targets.flatten().numpy()
    
    # Evaluate CatBoost
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    catboost_predictions = catboost_model.predict(X_test)
    
    # Calculate metrics
    gman_rmse = np.sqrt(mean_squared_error(test_targets_flat, gman_predictions))
    gman_mae = mean_absolute_error(test_targets_flat, gman_predictions)
    
    catboost_rmse = np.sqrt(mean_squared_error(y_test, catboost_predictions))
    catboost_mae = mean_absolute_error(y_test, catboost_predictions)
    
    return {
        'GMAN': {
            'RMSE': gman_rmse,
            'MAE': gman_mae,
            'Predictions': gman_predictions
        },
        'CatBoost': {
            'RMSE': catboost_rmse,
            'MAE': catboost_mae,
            'Predictions': catboost_predictions
        }
    }

# Plot results
def plot_results(results, test_df):
    plt.figure(figsize=(12, 6))
    
    # Get a specific node to visualize
    node_id = 0
    node_data = test_df[test_df['node_id'] == node_id].sort_values('time_step')
    time_steps = node_data['time_step'].values
    actual_values = node_data['target'].values
    
    # Get predictions for the specific node
    gman_preds = results['GMAN']['Predictions'][node_data.index]
    catboost_preds = results['CatBoost']['Predictions'][node_data.index]
    
    plt.plot(time_steps, actual_values, 'k-', label='Ground Truth')
    plt.plot(time_steps, gman_preds, 'b-', label=f'GMAN (RMSE: {results["GMAN"]["RMSE"]:.4f})')
    plt.plot(time_steps, catboost_preds, 'r-', label=f'CatBoost (RMSE: {results["CatBoost"]["RMSE"]:.4f})')
    
    plt.title(f'Chickenpox Predictions for Node {node_id}')
    plt.xlabel('Time Step')
    plt.ylabel('Chickenpox Cases')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('chickenpox_predictions.png')
    plt.show()

# Main execution
print("Training GMAN model...")
gman_losses = train_gman(gman_model, optimizer, features, targets, edge_indices, edge_weights, epochs=100)

print("Training CatBoost model...")
catboost_model = train_catboost(catboost_model, train_df)

print("Evaluating models...")
results = evaluate_models(gman_model, catboost_model, test_features, test_targets, test_edge_indices, test_edge_weights, test_df)

print("\nResults:")
print(f"GMAN - RMSE: {results['GMAN']['RMSE']:.4f}, MAE: {results['GMAN']['MAE']:.4f}")
print(f"CatBoost - RMSE: {results['CatBoost']['RMSE']:.4f}, MAE: {results['CatBoost']['MAE']:.4f}")

print("\nPlotting results...")
plot_results(results, test_df)

# Feature importance analysis for CatBoost
feature_importance = catboost_model.get_feature_importance()
feature_names = test_df.drop('target', axis=1).columns

plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
plt.title('CatBoost Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Generate final comparison report
comparison_df = pd.DataFrame({
    'Model': ['GMAN', 'CatBoost'],
    'RMSE': [results['GMAN']['RMSE'], results['CatBoost']['RMSE']],
    'MAE': [results['GMAN']['MAE'], results['CatBoost']['MAE']]
})

print("\nModel Comparison Report:")
print(comparison_df.to_string(index=False))