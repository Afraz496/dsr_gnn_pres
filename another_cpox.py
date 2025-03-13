import torch
import torch.nn.functional as F
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.nn.recurrent import DCRNN, GConvGRU
import pandas as pd
import numpy as np
import networkx as nx
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_cytoscape as cyto
import dash_leaflet as dl
import datetime
from sklearn.preprocessing import MinMaxScaler
import math

# Define an improved GNN model with attention and increased capacity
class ImprovedTemporalGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim=128, out_dim=1, dropout_rate=0.3):
        super(ImprovedTemporalGNN, self).__init__()
        # Increase model capacity with larger hidden dimensions
        self.recurrent1 = DCRNN(node_features, hidden_dim, 1)
        self.recurrent2 = GConvGRU(hidden_dim, hidden_dim, 2)  # Use different recurrent layer
        
        # Add multiple dropout layers for better regularization
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        
        # Add batch normalization for better stability
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim // 2)
        
        # Deeper MLP for prediction with residual connections
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear2 = torch.nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.linear3 = torch.nn.Linear(hidden_dim // 4, out_dim)
        
        # Add learnable seasonality embedding
        self.month_embedding = torch.nn.Embedding(12, 16)
        self.seasonal_linear = torch.nn.Linear(16, hidden_dim // 4)
        
    def forward(self, x, edge_index, edge_weight, month_idx=None):
        # Primary GNN layers
        h = self.recurrent1(x, edge_index, edge_weight)
        h = F.elu(h)  # ELU for better gradient flow
        h = self.dropout1(h)
        
        h_size = h.size(0)
        if h_size > 1:  # Only apply batch norm if we have more than one node
            h = self.batch_norm1(h)
        
        h = self.recurrent2(h, edge_index, edge_weight)
        h = F.elu(h)
        h = self.dropout2(h)
        
        # MLP prediction head
        h_res = h  # Save for residual connection
        h = self.linear1(h)
        h = F.elu(h)
        
        if h_size > 1:
            h = self.batch_norm2(h)
        
        h = self.linear2(h)
        h = F.elu(h)
        
        # Add seasonal information if provided
        if month_idx is not None:
            month_embed = self.month_embedding(month_idx)
            seasonal_features = self.seasonal_linear(month_embed)
            # Add seasonal features to each node
            seasonal_features = seasonal_features.expand(h.size(0), -1)
            h = h + seasonal_features
        
        # Final prediction layer
        h = self.linear3(h)
        
        return h

def get_dataset_snapshots(dataset):
    """Convert the iterable dataset to a list of snapshots"""
    return [snapshot for snapshot in dataset]

def normalize_coordinates(lat, lon, width=500, height=500):
    """Normalize latitude and longitude to fit in a given width and height"""
    min_lat, max_lat = 45.74, 48.58
    min_lon, max_lon = 16.11, 22.90

    x = (lon - min_lon) / (max_lon - min_lon) * width
    y = (max_lat - lat) / (max_lat - min_lat) * height
    return {'x': x, 'y': y}

def load_raw_data():
    """Load raw data with error handling"""
    try:
        df = pd.read_csv('hungary_chickenpox.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        # Create synthetic data for demo purposes if file not found
        print("Warning: Data file not found. Creating synthetic demo data.")
        dates = pd.date_range(start='2005-01-01', end='2015-12-31', freq='W-MON')
        counties = ['BUDAPEST','BARANYA','BACS','BEKES','BORSOD','CSONGRAD','FEJER','GYOR','HAJDU',
                    'HEVES','JASZ','KOMAROM','NOGRAD','PEST','SOMOGY','SZABOLCS','TOLNA','VAS','VESZPREM','ZALA']
        
        data = {}
        data['Date'] = dates
        
        # Generate synthetic seasonality patterns for each county
        for county in counties:
            base_cases = np.random.randint(50, 500)  # Base cases vary by county
            seasonality = np.sin(np.linspace(0, 10*np.pi, len(dates))) * 0.5 + 0.5  # Seasonal pattern
            trend = np.linspace(0, 0.5, len(dates))  # Slight upward trend
            noise = np.random.normal(0, 0.15, len(dates))  # Random noise
            
            # Combine factors and ensure minimum cases of 10
            cases = (base_cases * (1 + seasonality + trend + noise)).astype(int)
            cases = np.maximum(cases, 10)
            data[county] = cases
            
        return pd.DataFrame(data)

# County mapping as per CSV columns
csv_counties = ['BUDAPEST','BARANYA','BACS','BEKES','BORSOD','CSONGRAD','FEJER','GYOR','HAJDU',
                'HEVES','JASZ','KOMAROM','NOGRAD','PEST','SOMOGY','SZABOLCS','TOLNA','VAS','VESZPREM','ZALA']

# Mapping for visualization with more accurate coordinates
county_coordinates = {
    'BACS': (46.5935, 19.3547), 'BARANYA': (45.9910, 18.2232), 'BEKES': (46.7639, 21.0845),
    'BORSOD': (48.2286, 20.6180), 'CSONGRAD': (46.4167, 20.2500), 'FEJER': (47.1130, 18.4367),
    'GYOR': (47.6849, 17.2610), 'HAJDU': (47.5160, 21.6500), 'HEVES': (47.8500, 20.0833),
    'JASZ': (47.1667, 20.4167), 'KOMAROM': (47.6833, 18.3333), 'NOGRAD': (48.0000, 19.5000),
    'PEST': (47.5000, 19.3333), 'SOMOGY': (46.5833, 17.6667), 'SZABOLCS': (47.9000, 22.0000),
    'TOLNA': (46.5000, 18.5000), 'VAS': (47.0833, 16.5667), 'VESZPREM': (47.1000, 17.9000),
    'ZALA': (46.8333, 16.8333), 'BUDAPEST': (47.497913, 19.040236)
}

# Improved scaling function that's simpler for demo purposes
def create_simple_scalers():
    """Create simple scaling functions based on typical case ranges for demo purposes"""
    
    # Define typical ranges for each county (approximate for demo)
    county_ranges = {
        'BUDAPEST': (100, 1500),      # Largest city, highest cases
        'PEST': (80, 800),            # Large county surrounding Budapest
        'BORSOD': (50, 600),          # Larger county
        'HAJDU': (50, 500),
        'BACS': (40, 400),
        'CSONGRAD': (40, 350),
        'GYOR': (40, 350),
        'SZABOLCS': (35, 300),
        'FEJER': (35, 300),
        'JASZ': (30, 250),
        'BARANYA': (30, 250),
        'VESZPREM': (25, 220),
        'KOMAROM': (25, 200),
        'BEKES': (25, 200),
        'HEVES': (20, 180),
        'SOMOGY': (20, 180),
        'ZALA': (15, 150),
        'TOLNA': (15, 150),
        'VAS': (15, 140),
        'NOGRAD': (10, 120)           # Smallest county, fewer cases
    }
    
    # Create dictionaries for scaling
    min_values = {}
    max_values = {}
    
    for i, county in enumerate(csv_counties):
        min_values[i] = county_ranges[county][0]
        max_values[i] = county_ranges[county][1]
    
    # Simple scaling functions
    def scale_to_raw(scaled_value, county_idx):
        """Convert scaled model output (0-1) to meaningful case numbers"""
        # Ensure the scaled value is in 0-1 range
        scaled_value = max(0, min(1, scaled_value))
        # Map to the county's typical range
        return min_values[county_idx] + scaled_value * (max_values[county_idx] - min_values[county_idx])
    
    def raw_to_scaled(raw_value, county_idx):
        """Convert raw case numbers to model scale (0-1)"""
        # Ensure we don't divide by zero
        range_size = max(1, max_values[county_idx] - min_values[county_idx])
        # Scale to 0-1 range
        return (raw_value - min_values[county_idx]) / range_size
    
    return scale_to_raw, raw_to_scaled, min_values, max_values

# Enhanced Edge Creation - more sophisticated connectivity
def create_enhanced_edges(county_coords, population_weights=None):
    """Create edges based on proximity and population flows"""
    if population_weights is None:
        # Approximate population weights if not provided
        population_weights = {
            'BUDAPEST': 1.0,      # Capital city, highest weight
            'PEST': 0.9,          # Surrounds Budapest
            'BORSOD': 0.7,
            'HAJDU': 0.7,
            'BACS': 0.6,
            'CSONGRAD': 0.6,
            'GYOR': 0.6,
            'SZABOLCS': 0.5,
            'FEJER': 0.5,
            'BARANYA': 0.5,
            'JASZ': 0.4,
            'VESZPREM': 0.4,
            'KOMAROM': 0.4,
            'BEKES': 0.4,
            'HEVES': 0.4,
            'SOMOGY': 0.3,
            'ZALA': 0.3,
            'TOLNA': 0.3,
            'VAS': 0.3,
            'NOGRAD': 0.3,
        }
    
    edges = []
    edge_weights = []
    counties = list(county_coords.keys())
    
    # Direct connections based on actual geography of Hungary
    # These are the main connections between adjacent counties
    direct_connections = [
        ('BUDAPEST', 'PEST'), ('PEST', 'NOGRAD'), ('PEST', 'HEVES'), 
        ('PEST', 'JASZ'), ('PEST', 'BACS'), ('PEST', 'FEJER'), ('PEST', 'KOMAROM'),
        ('FEJER', 'KOMAROM'), ('FEJER', 'VESZPREM'), ('FEJER', 'TOLNA'), ('FEJER', 'BACS'),
        ('VESZPREM', 'GYOR'), ('VESZPREM', 'VAS'), ('VESZPREM', 'ZALA'), ('VESZPREM', 'SOMOGY'), ('VESZPREM', 'FEJER'),
        ('GYOR', 'KOMAROM'), ('GYOR', 'VAS'), ('GYOR', 'VESZPREM'),
        ('ZALA', 'VAS'), ('ZALA', 'SOMOGY'), ('ZALA', 'VESZPREM'),
        ('SOMOGY', 'TOLNA'), ('SOMOGY', 'BARANYA'), ('SOMOGY', 'ZALA'), ('SOMOGY', 'VESZPREM'),
        ('TOLNA', 'BARANYA'), ('TOLNA', 'BACS'), ('TOLNA', 'FEJER'), ('TOLNA', 'SOMOGY'),
        ('BARANYA', 'SOMOGY'), ('BARANYA', 'TOLNA'),
        ('BACS', 'TOLNA'), ('BACS', 'PEST'), ('BACS', 'JASZ'), ('BACS', 'CSONGRAD'),
        ('CSONGRAD', 'BACS'), ('CSONGRAD', 'BEKES'), ('CSONGRAD', 'JASZ'),
        ('BEKES', 'CSONGRAD'), ('BEKES', 'JASZ'), ('BEKES', 'HAJDU'),
        ('JASZ', 'PEST'), ('JASZ', 'HEVES'), ('JASZ', 'BORSOD'), ('JASZ', 'HAJDU'), 
        ('JASZ', 'BEKES'), ('JASZ', 'CSONGRAD'), ('JASZ', 'BACS'),
        ('HAJDU', 'JASZ'), ('HAJDU', 'BORSOD'), ('HAJDU', 'SZABOLCS'), ('HAJDU', 'BEKES'),
        ('BORSOD', 'SZABOLCS'), ('BORSOD', 'HAJDU'), ('BORSOD', 'JASZ'), ('BORSOD', 'HEVES'), ('BORSOD', 'NOGRAD'),
        ('HEVES', 'NOGRAD'), ('HEVES', 'BORSOD'), ('HEVES', 'JASZ'), ('HEVES', 'PEST'),
        ('NOGRAD', 'PEST'), ('NOGRAD', 'HEVES'), ('NOGRAD', 'BORSOD'),
        ('SZABOLCS', 'BORSOD'), ('SZABOLCS', 'HAJDU')
    ]
    
    # Create edges from direct connections with strong weights
    for src, dst in direct_connections:
        # Convert to indices
        src_idx = counties.index(src)
        dst_idx = counties.index(dst)
        
        # Calculate base weight based on population
        pop_factor = (population_weights[src] + population_weights[dst]) / 2
        
        # Add weighted edge (undirected, so add both directions)
        edges.append((src_idx, dst_idx))
        edge_weights.append(pop_factor * 0.8)  # Strong connection for adjacent counties
        
        edges.append((dst_idx, src_idx))
        edge_weights.append(pop_factor * 0.8)
    
    # Add special connections for Budapest to all counties
    # Budapest is the capital and has connections to all counties due to travel
    budapest_idx = counties.index('BUDAPEST')
    for i, county in enumerate(counties):
        if county != 'BUDAPEST':
            # Get coordinates
            bp_lat, bp_lon = county_coords['BUDAPEST']
            county_lat, county_lon = county_coords[county]
            
            # Calculate distance
            distance = math.sqrt((bp_lat - county_lat)**2 + (bp_lon - county_lon)**2)
            
            # Weight inversely proportional to distance and based on county population
            weight = (1.5 / (1 + distance)) * population_weights[county]
            
            # Add weighted connections to/from Budapest
            edges.append((budapest_idx, i))
            edge_weights.append(weight)
            
            edges.append((i, budapest_idx))
            edge_weights.append(weight)
    
    # Convert to tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    return edge_index, edge_weight

def extract_month_indices(time_labels):
    """Extract month indices (0-11) from time labels for seasonal modeling"""
    month_indices = []
    for label in time_labels:
        month_str = label.split(' ')[0]
        month_num = datetime.datetime.strptime(month_str, "%B").month - 1  # 0-indexed
        month_indices.append(month_num)
    return month_indices

def train_improved_gnn_model(dataset, edge_index=None, edge_weight=None):
    """Train an improved GNN model with better regularization and seasonality awareness"""
    snapshots = get_dataset_snapshots(dataset)
    total_steps = len(snapshots)
    
    # Generate time labels and extract month indices for seasonal modeling
    time_labels = generate_time_labels(start_year=2005, num_steps=total_steps)
    month_indices = extract_month_indices(time_labels)
    month_tensor = torch.tensor(month_indices, dtype=torch.long)
    
    # Use improved train/val/test split
    train_ratio, val_ratio = 0.7, 0.15
    train_split = int(total_steps * train_ratio)
    val_split = train_split + int(total_steps * val_ratio)
    
    # Use custom edge connections if provided, otherwise use the dataset's
    use_custom_edges = edge_index is not None and edge_weight is not None
    
    # Initialize improved model
    node_features = snapshots[0].x.shape[1]
    model = ImprovedTemporalGNN(node_features=node_features)
    
    # Use better optimizer settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=7, verbose=True, min_lr=1e-5
    )
    
    # Training loop with validation and enhanced early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 25  # Increased patience for better learning
    
    # Track losses for plotting
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(300):  # More epochs for better convergence
        # Training phase
        epoch_train_losses = []
        for i in range(train_split - 5):  # Use a window of 5 snapshots
            # Create a sequence of snapshots for temporal learning
            snapshot_sequence = []
            for j in range(5):  # Use 5 consecutive snapshots
                snapshot = snapshots[i+j]
                snapshot_sequence.append(snapshot)
            
            # Target is the next snapshot after the sequence
            target_snapshot = snapshots[i+5]
            y_true = target_snapshot.x[:, 0].reshape(-1, 1)
            
            # Get the latest snapshot in sequence for prediction
            latest_snapshot = snapshot_sequence[-1]
            
            optimizer.zero_grad()
            
            # Use custom edge structure if provided
            if use_custom_edges:
                x = latest_snapshot.x
                month_idx = month_tensor[i+4]  # Month of the latest snapshot
                out = model(x, edge_index, edge_weight, month_idx.unsqueeze(0))
            else:
                # Use edges from dataset
                month_idx = month_tensor[i+4]
                out = model(latest_snapshot.x, latest_snapshot.edge_index, 
                            latest_snapshot.edge_weight, month_idx.unsqueeze(0))
            
            # Calculate loss with L1 regularization for sparsity
            loss = F.mse_loss(out, y_true)
            l1_regularization = torch.tensor(0., requires_grad=True)
            for param in model.parameters():
                l1_regularization = l1_regularization + torch.norm(param, 1)
            
            total_loss = loss + 1e-5 * l1_regularization
            total_loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for i in range(train_split, val_split - 5):
                # Create validation sequence
                val_snapshot_sequence = []
                for j in range(5):
                    val_snapshot = snapshots[i+j]
                    val_snapshot_sequence.append(val_snapshot)
                
                target_snapshot = snapshots[i+5]
                y_true = target_snapshot.x[:, 0].reshape(-1, 1)
                
                # Get latest snapshot in sequence
                latest_snapshot = val_snapshot_sequence[-1]
                
                # Use custom edge structure if provided
                if use_custom_edges:
                    x = latest_snapshot.x
                    month_idx = month_tensor[i+4]
                    out = model(x, edge_index, edge_weight, month_idx.unsqueeze(0))
                else:
                    # Use edges from dataset
                    month_idx = month_tensor[i+4]
                    out = model(latest_snapshot.x, latest_snapshot.edge_index, 
                                latest_snapshot.edge_weight, month_idx.unsqueeze(0))
                
                loss = F.mse_loss(out, y_true)
                epoch_val_losses.append(loss.item())
        
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Generate predictions using the best model
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # For each timestep, we generate a prediction
        for i in range(total_steps):
            snapshot = snapshots[i]
            
            # Use custom edge structure if provided
            if use_custom_edges:
                month_idx = month_tensor[i]
                out = model(snapshot.x, edge_index, edge_weight, month_idx.unsqueeze(0))
            else:
                month_idx = month_tensor[i]
                out = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight, month_idx.unsqueeze(0))
            
            # Non-negative predictions
            pred = torch.clamp(out, min=0).numpy().flatten()
            predictions.append(pred)
    
    return predictions, train_losses, val_losses, model

def create_improved_dashboard(dataset, raw_data, csv_counties):
    """Create an enhanced dashboard with improved visualizations"""
    # Convert dataset to a list of snapshots for direct indexing
    snapshots = get_dataset_snapshots(dataset)
    total_steps = len(snapshots)
    
    # Generate time labels
    time_labels = generate_time_labels(start_year=2005, num_steps=total_steps)
    
    # Create enhanced edge structure based on geography and population
    counties = list(county_coordinates.keys())
    edge_index, edge_weight = create_enhanced_edges(county_coordinates)
    
    # Train model using enhanced edges
    scaled_predictions, train_losses, val_losses, trained_model = train_improved_gnn_model(
        dataset, edge_index, edge_weight
    )
    
    # Create scaling functions for demo-friendly output
    scale_to_raw, raw_to_scaled, min_values, max_values = create_simple_scalers()
    
    # Convert predictions to raw scale
    raw_predictions = []
    for time_idx, pred in enumerate(scaled_predictions):
        raw_pred = [scale_to_raw(val, county_idx) for county_idx, val in enumerate(pred)]
        raw_predictions.append(raw_pred)
    
    # Initialize Dash app
    app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
    
    # Normalize coordinates for network visualization
    positions = {county: normalize_coordinates(lat, lon) for county, (lat, lon) in county_coordinates.items()}
    
    # Create time slider with meaningful marks
    step_size = max(1, total_steps // 10)
    time_slider_marks = {i: time_labels[i] for i in range(0, total_steps, step_size) if i < total_steps}
    
    # Create app layout with improved UI
    app.layout = html.Div([
        html.H1("Hungary Chickenpox Cases - Advanced GNN Analysis", 
                style={'textAlign': 'center', 'marginBottom': 30, 'color': '#2C3E50'}),
        
        html.Div([
            html.Div([
                # Model performance metrics
                html.Div([
                    html.H3("Model Training Performance", style={'textAlign': 'center'}),
                    dcc.Graph(
                        id='training-graph',
                        figure={
                            'data': [
                                {'x': list(range(len(train_losses))), 'y': train_losses, 
                                 'type': 'line', 'name': 'Training Loss'},
                                {'x': list(range(len(val_losses))), 'y': val_losses, 
                                 'type': 'line', 'name': 'Validation Loss'}
                            ],
                            'layout': {
                                'title': 'Loss Curves',
                                'xaxis': {'title': 'Epoch'},
                                'yaxis': {'title': 'Loss (MSE)'},
                                'legend': {'x': 0, 'y': 1},
                                'margin': {'l': 50, 'r': 50, 'b': 50, 't': 70}
                            }
                        }
                    )
                ], style={'marginBottom': 30}),
                
                # Map view
                html.Div([
                    html.H3("Geographic Case Distribution", style={'textAlign': 'center'}),
                    dl.Map(
                        [
                            dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"),
                            dl.LayerGroup(id='markers')
                        ],
                        center=[47.1625, 19.5033],
                        zoom=7,
                        style={'width': '100%', 'height': '400px'}, 
                        id='map'
                    )
                ]),
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                # Network graph
                html.Div([
                    html.H3("County Disease Spread Network", style={'textAlign': 'center'}),
                    cyto.Cytoscape(
                        id='cytoscape-graph',
                        elements=[],
                        layout={'name': 'preset'},
                        style={'width': '100%', 'height': '400px'},
                        stylesheet=[
                            {
                                'selector': 'node',
                                'style': {
                                    'label': 'data(label)',
                                    'background-color': 'data(color)',
                                    'font-size': '14px',
                                    'text-wrap': 'wrap',
                                    'width': 'data(size)',
                                    'height': 'data(size)',
                                    'padding': '8px',
                                    'text-valign': 'center',
                                    'text-halign': 'center',
                                    'text-outline-width': 1,
                                    'text-outline-color': '#FFFFFF'
                                }
                            },
                            {
                                'selector': 'edge',
                                'style': {
                                    'width': 'data(weight)',
                                    'line-color': '#888',
                                    'curve-style': 'bezier',
                                    'opacity': 0.7
                                }
                            }
                        ]
                    ),
                ]),
                
                # Advanced case comparison
                html.Div([
                    html.H3("Model Performance Analysis", style={'textAlign': 'center'}),
                    html.Div(id='model-metrics', style={'marginBottom': 10, 'fontWeight': 'bold', 'textAlign': 'center'}),
                ], style={'marginTop': 20}),
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ]),
        
        # Improved time slider
        html.Div([
            html.Label("Time Period", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='time-slider',
                min=0, max=total_steps - 1, step=1, value=0,
                marks=time_slider_marks,
                updatemode='drag'
            ),
        ], style={'marginTop': 20, 'marginBottom': 30}),
        
        # Detailed county comparison
        html.Div([
            html.H3("County Case Comparison - Actual vs Predicted", style={'textAlign': 'center', 'marginBottom': 20}),
            dcc.Graph(id='comparison-graph')
        ]),
        
        # County selection for detailed analysis
        html.Div([
            html.H3("Single County Time Series Analysis", style={'textAlign': 'center', 'marginBottom': 20}),
            html.Div([
                html.Label("Select County:", style={'fontWeight': 'bold', 'marginRight': 10}),
                dcc.Dropdown(
                    id='county-dropdown',
                    options=[{'label': county, 'value': i} for i, county in enumerate(csv_counties)],
                    value=0,  # Default to first county (Budapest)
                    style={'width': '300px'}
                ),
            ], style={'marginBottom': 20, 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),
            dcc.Graph(id='county-time-series')
        ]),
        
        html.Footer([
            html.Hr(),
            html.P("Hungary Chickenpox GNN Analysis - © 2025", 
                   style={'textAlign': 'center', 'marginTop': 20, 'color': '#777'})
        ])
    ])
    
    # Define callback for time slider to update map markers
    @app.callback(
        Output('markers', 'children'),
        [Input('time-slider', 'value')]
    )
    def update_map_markers(selected_time):
        # Get predicted values for current time step
        pred_values = raw_predictions[selected_time]
        
        # Normalize for color intensity (0-100%)
        max_value = max(pred_values) * 1.2  # Give some headroom
        
        # Create markers for each county
        markers = []
        for i, county in enumerate(counties):
            # Get county coordinates
            lat, lon = county_coordinates[county]
            
            # Get predicted value and calculate radius & color
            value = pred_values[i]
            norm_value = min(1.0, value / max_value)
            
            # Color from yellow (low) to red (high)
            r = int(255)
            g = int(255 * (1 - norm_value * 0.8))
            b = int(50 * (1 - norm_value))
            color = f'rgb({r}, {g}, {b})'
            
            # Radius based on case numbers (min 8, max 25)
            radius = 8 + norm_value * 17
            
            # Create circle marker
            marker = dl.CircleMarker(
                center=[lat, lon],
                radius=radius,
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=1,
                children=[dl.Tooltip(f"{county}: {int(value)} predicted cases")]
            )
            markers.append(marker)
        
        return markers
    
    # Define callback for time slider to update network graph
    @app.callback(
        Output('cytoscape-graph', 'elements'),
        [Input('time-slider', 'value')]
    )
    def update_network(selected_time):
        # Get predicted values for current time step
        pred_values = raw_predictions[selected_time]
        
        # Normalize for node size and color
        max_value = max(pred_values) * 1.2  # Give some headroom
        
        # Create nodes
        nodes = []
        for i, county in enumerate(counties):
            # Get county position
            pos = positions[county]
            
            # Get predicted value and normalize
            value = pred_values[i]
            norm_value = min(1.0, value / max_value)
            
            # Color from yellow (low) to red (high)
            r = int(255)
            g = int(255 * (1 - norm_value * 0.8))
            b = int(50 * (1 - norm_value))
            color = f'rgb({r}, {g}, {b})'
            
            # Node size based on case numbers
            size = 20 + norm_value * 30
            
            # Create node
            nodes.append({
                'data': {
                    'id': county,
                    'label': f"{county}\n({int(value)})",
                    'color': color,
                    'size': size
                },
                'position': {'x': pos['x'], 'y': pos['y']}
            })
        
        # Create edges
        edges = []
        edge_index_np = edge_index.numpy()
        edge_weight_np = edge_weight.numpy()
        
        for i in range(edge_index_np.shape[1]):
            src_idx = edge_index_np[0, i]
            dst_idx = edge_index_np[1, i]
            weight = edge_weight_np[i]
            
            # Skip weak connections for cleaner visualization
            if weight < 0.3:
                continue
                
            # Create edge
            edges.append({
                'data': {
                    'source': counties[src_idx],
                    'target': counties[dst_idx],
                    'weight': weight * 3  # Scale weight for visibility
                }
            })
        
        # Combine nodes and edges
        elements = nodes + edges
        return elements
    
    # Define callback for comparison graph
    @app.callback(
        Output('comparison-graph', 'figure'),
        [Input('time-slider', 'value')]
    )
    def update_comparison(selected_time):
        # Get raw data for the selected time
        time_label = time_labels[selected_time]
        
        # Extract date from time label (format: "Month Year (Week X)")
        month_year = ' '.join(time_label.split(' ')[:2])
        date_str = month_year + " 1"  # Add day for parsing
        target_date = datetime.datetime.strptime(date_str, "%B %Y %d")
        
        # Find closest date in raw data
        if 'Date' in raw_data.columns:
            closest_date_idx = (raw_data['Date'] - target_date).abs().idxmin()
            actual_data = raw_data.iloc[closest_date_idx].drop('Date').values
        else:
            # If no date column, just use the predicted values
            actual_data = np.zeros(len(csv_counties))
        
        # Get predicted values
        predicted_data = raw_predictions[selected_time]
        
        # Create comparison bar chart
        figure = {
            'data': [
                {'x': csv_counties, 'y': actual_data, 'type': 'bar', 'name': 'Actual Cases'},
                {'x': csv_counties, 'y': predicted_data, 'type': 'bar', 'name': 'Predicted Cases'}
            ],
            'layout': {
                'title': f"Chickenpox Cases Comparison for {time_label}",
                'xaxis': {'title': 'County'},
                'yaxis': {'title': 'Number of Cases'},
                'barmode': 'group',
                'legend': {'x': 0, 'y': 1.1, 'orientation': 'h'},
                'margin': {'l': 50, 'r': 50, 'b': 100, 't': 70},
                'height': 500
            }
        }
        return figure
    
    # Define callback for model metrics
    @app.callback(
        Output('model-metrics', 'children'),
        [Input('time-slider', 'value')]
    )
    def update_metrics(selected_time):
        # Get raw data for the selected time
        time_label = time_labels[selected_time]
        
        # Extract date from time label (format: "Month Year (Week X)")
        month_year = ' '.join(time_label.split(' ')[:2])
        date_str = month_year + " 1"  # Add day for parsing
        target_date = datetime.datetime.strptime(date_str, "%B %Y %d")
        
        # Find closest date in raw data
        if 'Date' in raw_data.columns:
            closest_date_idx = (raw_data['Date'] - target_date).abs().idxmin()
            actual_data = raw_data.iloc[closest_date_idx].drop('Date').values
        else:
            # If no date column, just use zeros
            actual_data = np.zeros(len(csv_counties))
        
        # Get predicted values
        predicted_data = raw_predictions[selected_time]
        
        # Calculate metrics
        mse = np.mean((actual_data - predicted_data) ** 2)
        mae = np.mean(np.abs(actual_data - predicted_data))
        
        # Return metrics text
        return [
            html.Div(f"Mean Squared Error: {mse:.2f}"),
            html.Div(f"Mean Absolute Error: {mae:.2f}")
        ]
    
    # Define callback for county time series
    @app.callback(
        Output('county-time-series', 'figure'),
        [Input('county-dropdown', 'value')]
    )
    def update_county_series(selected_county):
        # Extract county name
        county_name = csv_counties[selected_county]
        
        # Extract time series data for this county
        predicted_series = [pred[selected_county] for pred in raw_predictions]
        
        # Extract actual data if available
        if 'Date' in raw_data.columns:
            # Get actual time series
            actual_series = raw_data[county_name].values
            
            # If lengths differ, truncate to match
            min_len = min(len(actual_series), len(predicted_series))
            actual_series = actual_series[:min_len]
            predicted_series = predicted_series[:min_len]
            
            # Create time labels for x-axis
            x_labels = time_labels[:min_len]
            
            # Create figure with both series
            figure = {
                'data': [
                    {'x': x_labels, 'y': actual_series, 'type': 'line', 'name': 'Actual Cases'},
                    {'x': x_labels, 'y': predicted_series, 'type': 'line', 'name': 'Predicted Cases'}
                ],
                'layout': {
                    'title': f"Time Series for {county_name}",
                    'xaxis': {'title': 'Time Period'},
                    'yaxis': {'title': 'Number of Cases'},
                    'legend': {'x': 0, 'y': 1.1, 'orientation': 'h'},
                    'margin': {'l': 50, 'r': 50, 'b': 100, 't': 70},
                    'height': 400
                }
            }
        else:
            # Just show predictions
            figure = {
                'data': [
                    {'x': time_labels, 'y': predicted_series, 'type': 'line', 'name': 'Predicted Cases'}
                ],
                'layout': {
                    'title': f"Predicted Time Series for {county_name}",
                    'xaxis': {'title': 'Time Period'},
                    'yaxis': {'title': 'Number of Cases'},
                    'margin': {'l': 50, 'r': 50, 'b': 100, 't': 70},
                    'height': 400
                }
            }
        
        return figure
    
    return app

def generate_time_labels(start_year=2005, num_steps=570):
    """Generate time labels for the dataset"""
    time_labels = []
    start_date = datetime.datetime(start_year, 1, 1)
    
    for i in range(num_steps):
        # Each step is approximately a week
        current_date = start_date + datetime.timedelta(weeks=i)
        month_name = current_date.strftime("%B")
        year = current_date.year
        week = (i % 52) + 1
        
        label = f"{month_name} {year} (Week {week})"
        time_labels.append(label)
    
    return time_labels

def main():
    """Main function to run the application"""
    print("Loading Hungary chickenpox dataset...")
    
    # Try to load the real dataset first
    try:
        loader = ChickenpoxDatasetLoader()
        dataset = loader.get_dataset()
        print("Successfully loaded PyTorch Geometric Temporal dataset")
    except Exception as e:
        print(f"Error loading dataset from PyTorch Geometric: {e}")
        print("Creating synthetic dataset instead...")
        # Implement synthetic dataset creation if needed
        dataset = None
    
    # Load raw data
    print("Loading raw chickenpox data...")
    try:
        raw_data = load_raw_data()
        print(f"Successfully loaded raw data with {raw_data.shape[0]} records")
    except Exception as e:
        print(f"Error loading raw data: {e}")
        print("Continuing with limited functionality...")
        raw_data = pd.DataFrame()
    
    # Create and run the dashboard
    print("Creating dashboard application...")
    app = create_improved_dashboard(dataset, raw_data, csv_counties)
    
    print("Starting server. Please open http://127.0.0.1:8050/ in your web browser")
    app.run_server(debug=True)

if __name__ == "__main__":
    main()