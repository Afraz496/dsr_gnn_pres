import torch
import torch_geometric_temporal
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric.data import Data
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
import torch.optim as optim
from torch_geometric_temporal.signal import temporal_signal_split
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import dense_to_sparse
import numpy as np
import pandas as pd
import json
import argparse
import os
import yaml
import sys
import dashviz
import logging
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.nn import LSTM

def load_custom_data(data, distance_matrix, normalize=False):
    """
    Parameters
    ----------
    data : pd.DataFrame
        Temporally ordered node features and predictions.
    distance_matrix : pd.DataFrame
        An N x N matrix of spatial connections.
    normalize : bool
        Whether to normalize the features and targets (default is True).
    
    Returns
    -------
    dataset : StaticGraphTemporalSignal
        A torch geometric temporal Static Graph Temporal Signal object
    num_nodes : int 
        Number of nodes in the Graph
    num_features : int
        Number of features per Node
    num_time_steps : int
        Total time period of the data.
    feature_scaler : StandardScaler
        The scaler used for normalizing features.
    target_scaler : StandardScaler
        The scaler used for normalizing targets.
    """
    # Get basic dimensions
    num_nodes = len(distance_matrix)
    num_time_steps = data[date_col].nunique()
    num_features = len(data.drop(columns=[date_col, prediction_col, LHA_ID]).columns)

    # Convert features to numpy array and reshape
    features = data.drop(columns=[date_col, prediction_col, LHA_ID]).to_numpy()
    features = features.reshape(num_time_steps, num_nodes, -1).astype(np.float32)

    # Normalize features if required
    if normalize:
        feature_scaler = StandardScaler()
        features = feature_scaler.fit_transform(features.reshape(-1, num_features))
        features = features.reshape(num_time_steps, num_nodes, num_features)
    else:
        feature_scaler = None

    # Convert target to numpy array and reshape
    targets = data[prediction_col].to_numpy()
    targets = targets.reshape(num_time_steps, num_nodes).astype(np.float32)
    # Normalize targets if required
    if normalize:
        logger.info("Before scaling (our prediction):")
        logger.info(f"Target min: {targets.min()}, max: {targets.max()}")
        target_scaler = MinMaxScaler(feature_range=(0, 396))
    # Reshape to (n_samples, 1) for fitting
        targets_flat = targets.reshape(-1, 1)
        print(f'Flattened target shape: {targets_flat.shape}')
        targets_scaled = target_scaler.fit_transform(targets_flat)
        # Reshape back to original dimensions
        targets = targets_scaled.reshape(num_time_steps, num_nodes).astype(np.float32)
        print(f'Target shape after reshaping: {targets.shape}')
        logger.info("After scaling:")
        logger.info(f"Target min: {targets.min()}, max: {targets.max()}")
    else:
        target_scaler = None
    
    # Convert adjacency matrix to edge list
    adjacency_matrix = distance_matrix.values.astype(np.float32)
    # Convert to torch tensor temporarily for dense_to_sparse operation
    adj_tensor = torch.tensor(adjacency_matrix)
    edge_index, edge_weight = dense_to_sparse(adj_tensor)
    
    # Convert back to numpy arrays
    edge_index = edge_index.numpy().astype(np.int64)  # Edge indices should be integers
    edge_weight = edge_weight.numpy().astype(np.float32)
    
    num_nodes = features.shape[1]
    connected_nodes = np.unique(edge_index.flatten())
    print(f"Number of connected nodes: {len(connected_nodes)} / {num_nodes}")
    print(f'Targets Shape: {targets.shape}')
    # Create temporal signal dataset
    dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weight,
        features=features,
        targets=targets
    )
    return dataset, num_nodes, num_features, num_time_steps, feature_scaler, target_scaler