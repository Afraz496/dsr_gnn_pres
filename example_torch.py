from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from tqdm import tqdm
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load Dataset
loader = ChickenpoxDatasetLoader()
dataset = loader.get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

# Simplified and Optimized DCRNN Model
class OptimizedRecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(OptimizedRecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 64, 1)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.linear(h)
        return h

model = OptimizedRecurrentGCN(node_features=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=50, mode='triangular2')

best_loss = float('inf')
patience, patience_counter = 30, 0

# Training Loop with Early Stopping and Scheduler
model.train()
for epoch in tqdm(range(500)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss = F.mse_loss(y_hat.squeeze(), snapshot.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        cost += loss.item()

    scheduler.step()
    cost /= (time + 1)

    if cost < best_loss:
        best_loss = cost
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Evaluate Optimized DCRNN
model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    cost += F.mse_loss(y_hat.squeeze(), snapshot.y).item()
cost /= (time + 1)
print("Optimized DCRNN MSE: {:.4f}".format(cost))

# Prepare Data for CatBoost
def flatten_dataset(dataset):
    X, y = [], []
    for snapshot in dataset:
        X.append(snapshot.x.numpy())
        y.append(snapshot.y.numpy())
    return np.vstack(X), np.vstack(y).flatten()

X_train, y_train = flatten_dataset(train_dataset)
X_test, y_test = flatten_dataset(test_dataset)

# Train CatBoost
cat_model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0)
cat_model.fit(X_train, y_train)
y_pred = cat_model.predict(X_test)
cat_mse = mean_squared_error(y_test, y_pred)
print(f"CatBoost MSE: {cat_mse:.4f}")

# Comparison
if cat_mse < cost:
    print("CatBoost performed better than Optimized DCRNN.")
elif cat_mse > cost:
    print("Optimized DCRNN performed better than CatBoost.")
else:
    print("Both models performed equally.")
