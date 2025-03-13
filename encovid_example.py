import json
import ssl
import urllib.request
import numpy as np
import torch
from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# Initialize loader
loader = EnglandCovidDatasetLoader()

# Get dataset with a lag of 8 days (you can change the lag)
lags = 8
dataset = loader.get_dataset(lags)

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class TemporalGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TemporalGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Initialize the Temporal GNN model
model = TemporalGNN(input_dim=8, hidden_dim=16, output_dim=1)  # Adjust dimensions as needed

# Prepare CatBoost tabular features and labels
features = np.array(dataset.features)  # Assuming dataset.features is a 3D array
targets = np.array(dataset.targets)    # Assuming dataset.targets is a 2D array

# Reshape or flatten if necessary for CatBoost (usually it works with 2D arrays)
X_catboost = features.reshape(-1, features.shape[-1])  # Flattening if needed
y_catboost = targets.flatten()  # Flatten target if needed

# Train-test split for CatBoost
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_catboost, y_catboost, test_size=0.2, shuffle=False)

# Train CatBoost model
catboost_model = CatBoostRegressor(iterations=1000, depth=10, learning_rate=0.05, loss_function='RMSE')
catboost_model.fit(X_train, y_train)

# Make predictions with CatBoost
catboost_preds = catboost_model.predict(X_test)

# Prepare to store GNN predictions
gnn_preds_list = []

# Assuming the test period corresponds to the last portion of the time steps
test_start_idx = len(dataset.features) - len(y_test)  # Adjust based on how you define the test period

# Iterate over the time steps for inference (using dataset.edge_index and dataset.features)
for time_step in range(test_start_idx, len(dataset.features)):
    # Get the current time step's edges, edge weights, and features
    edge_index = dataset.edge_indices[time_step]
    edge_weight = dataset.edge_weights[time_step]  # Optional, only if used in model
    x = torch.tensor(dataset.features[time_step], dtype=torch.float)

    # Convert edge_index to a tensor if it is not already in the correct format
    edge_index = torch.tensor(dataset.edge_indices[time_step], dtype=torch.long)

    # Forward pass through the model
    gnn_preds = model(x, edge_index)

    # Store the predictions (flattening if necessary)
    gnn_preds_list.append(gnn_preds.detach().numpy().flatten())

# Convert the list to a numpy array for easier handling
gnn_preds_array = np.array(gnn_preds_list)

# Ensure predictions match the test set
assert len(y_test) == len(gnn_preds_array.flatten()), \
    f"Shape mismatch: y_test has {len(y_test)} samples, but predictions have {len(gnn_preds_array.flatten())} samples."

# Calculate MSE for both models
gnn_mse = mean_squared_error(y_test, gnn_preds_array.flatten())  # PyTorch model
catboost_mse = mean_squared_error(y_test, catboost_preds)  # CatBoost model

# Print the results
print(f"PyTorch GNN Model MSE: {gnn_mse}")
print(f"CatBoost Model MSE: {catboost_mse}")


# Visualization
import matplotlib.pyplot as plt

plt.plot(y_test, label='True Values', linestyle='--')
plt.plot(gnn_preds_array.flatten(), label='GNN Predictions', linestyle='-')
plt.plot(catboost_preds, label='CatBoost Predictions', linestyle='-.')

plt.legend()
plt.title("Comparison of PyTorch GNN vs CatBoost")
plt.show()
