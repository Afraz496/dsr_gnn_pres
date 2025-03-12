import torch
import torch_geometric_temporal
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
import pandas as pd
import numpy as np
import networkx as nx
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_cytoscape as cyto
import dash_leaflet as dl
import datetime

# Define a simple GNN model for predictions
class EnhancedRecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim=64):
        super(EnhancedRecurrentGCN, self).__init__()
        self.recurrent1 = DCRNN(node_features, hidden_dim, 1)
        self.recurrent2 = DCRNN(hidden_dim, hidden_dim // 2, 1)
        self.dropout = torch.nn.Dropout(0.2)
        self.linear1 = torch.nn.Linear(hidden_dim // 2, 16)
        self.linear2 = torch.nn.Linear(16, 1)
        
    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.recurrent2(h, edge_index, edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.linear2(h)
        return h

def normalize_coordinates(lat, lon, width=500, height=500):
    # Normalize latitude and longitude to fit in a given width and height
    min_lat, max_lat = 45.74, 48.58
    min_lon, max_lon = 16.11, 22.90

    x = (lon - min_lon) / (max_lon - min_lon) * width
    y = (max_lat - lat) / (max_lat - min_lat) * height
    return {'x': x, 'y': y}

# Load raw data
def load_raw_data():
    df = pd.read_csv('hungary_chickenpox.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# County mapping as per CSV columns
csv_counties = ['BUDAPEST','BARANYA','BACS','BEKES','BORSOD','CSONGRAD','FEJER','GYOR','HAJDU',
                'HEVES','JASZ','KOMAROM','NOGRAD','PEST','SOMOGY','SZABOLCS','TOLNA','VAS','VESZPREM','ZALA']

# Mapping for visualization
county_coordinates = {
    'BACS': (46.5935, 19.3547), 'BARANYA': (45.9910, 18.2232), 'BEKES': (46.7639, 21.0845),
    'BORSOD': (48.2286, 20.6180), 'CSONGRAD': (46.4167, 20.2500), 'FEJER': (47.1130, 18.4367),
    'GYOR': (47.6849, 17.2610), 'HAJDU': (47.5160, 21.6500), 'HEVES': (47.8500, 20.0833),
    'JASZ': (47.1667, 20.4167), 'KOMAROM': (47.6833, 18.3333), 'NOGRAD': (48.0000, 19.5000),
    'PEST': (47.5000, 19.3333), 'SOMOGY': (46.5833, 17.6667), 'SZABOLCS': (47.9000, 22.0000),
    'TOLNA': (46.5000, 18.5000), 'VAS': (47.0833, 16.5667), 'VESZPREM': (47.1000, 17.9000),
    'ZALA': (46.8333, 16.8333), 'BUDAPEST': (47.497913, 19.040236)
}

def get_dataset_snapshots(dataset):
    """Convert the iterable dataset to a list of snapshots"""
    snapshots = []
    for snapshot in dataset:
        snapshots.append(snapshot)
    return snapshots

def create_scalers(dataset, raw_data, csv_counties):
    """
    Create scaling functions to convert between scaled and raw data values.
    
    Args:
        dataset: The PyTorch Geometric Temporal dataset with scaled values
        raw_data: DataFrame containing unscaled raw data
        csv_counties: List of county names
    
    Returns:
        scale_up_fn: Function to convert scaled predictions to raw scale
        scale_down_fn: Function to convert raw values to the dataset scale
    """
    import numpy as np
    
    # Get first few snapshots from the dataset
    snapshots = get_dataset_snapshots(dataset)
    
    # Extract scaled values for each county
    scaled_values = []
    for i, snapshot in enumerate(snapshots[:10]):  # Use first 10 snapshots
        scaled_values.append(snapshot.x[:, 0].numpy())
    
    scaled_matrix = np.array(scaled_values)
    
    # Extract the corresponding raw values
    # Create a mapping of dates to use for comparison
    time_labels = generate_time_labels(start_year=2005, num_steps=len(snapshots))
    
    # Extract dates for the first 10 snapshots
    dates = []
    for i in range(10):
        time_str = time_labels[i].split(' ')
        month, year = time_str[0], int(time_str[1])
        month_num = datetime.datetime.strptime(month, "%B").month
        dates.append((year, month_num))
    
    # Get raw values for these dates
    raw_values = []
    for year, month in dates:
        filtered_data = raw_data[
            (raw_data['Date'].dt.year == year) & 
            (raw_data['Date'].dt.month == month)
        ]
        
        if not filtered_data.empty:
            raw_values.append(filtered_data[csv_counties].values[0])
        else:
            # If no data for this date, use zeros
            raw_values.append(np.zeros(len(csv_counties)))
    
    raw_matrix = np.array(raw_values)
    
    # Calculate scaling factors for each county
    scaling_factors = []
    scaling_offsets = []
    
    for county_idx in range(len(csv_counties)):
        scaled_county_values = scaled_matrix[:, county_idx]
        raw_county_values = raw_matrix[:, county_idx]
        
        # Avoid division by zero and use only non-zero entries
        valid_indices = (scaled_county_values != 0) & (raw_county_values != 0)
        
        if np.sum(valid_indices) > 0:
            # For each county, calculate average ratio between raw and scaled
            ratios = raw_county_values[valid_indices] / scaled_county_values[valid_indices]
            factor = np.median(ratios)  # Use median to be robust against outliers
            
            # Calculate typical offset (in case scaling is not just multiplication)
            offsets = raw_county_values[valid_indices] - (scaled_county_values[valid_indices] * factor)
            offset = np.median(offsets)
        else:
            # Default values if no valid data points
            factor = 1.0
            offset = 0.0
            
        scaling_factors.append(factor)
        scaling_offsets.append(offset)
    
    # Create scaling functions
    def scale_up_fn(scaled_predictions, county_idx=None):
        """Convert scaled predictions to raw scale"""
        if county_idx is not None:
            # Scale for a specific county
            return scaled_predictions * scaling_factors[county_idx] + scaling_offsets[county_idx]
        else:
            # Scale for all counties
            return np.array([
                scaled_predictions[i] * scaling_factors[i] + scaling_offsets[i]
                for i in range(len(scaling_factors))
            ])
    
    def scale_down_fn(raw_values, county_idx=None):
        """Convert raw values to the dataset scale"""
        if county_idx is not None:
            # Only if scaling factor is not zero to avoid division by zero
            if scaling_factors[county_idx] != 0:
                return (raw_values - scaling_offsets[county_idx]) / scaling_factors[county_idx]
            else:
                return raw_values
        else:
            # Scale for all counties
            return np.array([
                (raw_values[i] - scaling_offsets[i]) / scaling_factors[i] 
                if scaling_factors[i] != 0 else raw_values[i]
                for i in range(len(scaling_factors))
            ])
    
    return scale_up_fn, scale_down_fn, scaling_factors, scaling_offsets


def train_gnn_model_with_scaling(dataset, raw_data, csv_counties):
    """Train a GNN model on the dataset with validation and scale predictions"""
    # First get the scalers
    scale_up_fn, scale_down_fn, factors, offsets = create_scalers(dataset, raw_data, csv_counties)
    
    # Print scaling factors for debugging
    print("Scaling factors:", factors)
    print("Scaling offsets:", offsets)
    
    snapshots = get_dataset_snapshots(dataset)
    train_split = int(len(snapshots) * 0.7)
    val_split = int(len(snapshots) * 0.85)
    
    # Initialize enhanced model
    node_features = snapshots[0].x.shape[1]
    model = EnhancedRecurrentGCN(node_features=node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop with validation
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 15  # Early stopping
    
    model.train()
    for epoch in range(200):  # More epochs
        # Training phase
        train_losses = []
        for i in range(train_split - 1):  # Use t to predict t+1
            snapshot = snapshots[i]
            y_true = snapshots[i+1].x[:, 0].reshape(-1, 1)  # Target is next snapshot's feature
            
            optimizer.zero_grad()
            out = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
            loss = F.mse_loss(out, y_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(train_split, val_split - 1):
                snapshot = snapshots[i]
                y_true = snapshots[i+1].x[:, 0].reshape(-1, 1)
                
                out = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
                loss = F.mse_loss(out, y_true)
                val_losses.append(loss.item())
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
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
    
    # Generate predictions (scaled)
    model.eval()
    scaled_predictions = []
    raw_scale_predictions = []
    
    with torch.no_grad():
        for snapshot in snapshots:
            pred = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
            # Ensure predictions are non-negative
            pred = torch.clamp(pred, min=0)
            
            # Get scaled predictions
            scaled_pred = pred.numpy().flatten()
            scaled_predictions.append(scaled_pred)
            
            # Transform to raw scale
            raw_pred = scale_up_fn(scaled_pred)
            raw_scale_predictions.append(raw_pred)
    
    return scaled_predictions, raw_scale_predictions, factors, offsets

def train_gnn_model(dataset):
    """Train a more sophisticated GNN model on the dataset with validation"""
    snapshots = get_dataset_snapshots(dataset)
    train_split = int(len(snapshots) * 0.7)
    val_split = int(len(snapshots) * 0.85)
    
    # Initialize enhanced model
    node_features = snapshots[0].x.shape[1]
    model = EnhancedRecurrentGCN(node_features=node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop with validation
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 15  # Early stopping
    
    model.train()
    for epoch in range(200):  # More epochs
        # Training phase
        train_losses = []
        for i in range(train_split - 1):  # Use t to predict t+1
            snapshot = snapshots[i]
            y_true = snapshots[i+1].x[:, 0].reshape(-1, 1)  # Target is next snapshot's feature
            
            optimizer.zero_grad()
            out = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
            loss = F.mse_loss(out, y_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(train_split, val_split - 1):
                snapshot = snapshots[i]
                y_true = snapshots[i+1].x[:, 0].reshape(-1, 1)
                
                out = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
                loss = F.mse_loss(out, y_true)
                val_losses.append(loss.item())
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
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
    
    # Generate predictions
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for snapshot in snapshots:
            pred = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
            # Ensure predictions are non-negative
            pred = torch.clamp(pred, min=0)
            predictions.append(pred.numpy().flatten())
    
    return predictions

def create_dashboard(dataset, raw_data, csv_counties):
    # Convert dataset to a list of snapshots for direct indexing
    snapshots = get_dataset_snapshots(dataset)
    total_steps = len(snapshots)
    
    # Train model and get predictions - both scaled and raw
    scaled_predictions, raw_scale_predictions, scaling_factors, scaling_offsets = train_gnn_model_with_scaling(dataset, raw_data, csv_counties)
    
    print("First few predictions (scaled):", scaled_predictions[0][:5])
    print("First few predictions (raw scale):", raw_scale_predictions[0][:5])
    
    app = dash.Dash(__name__)
    time_labels = generate_time_labels(start_year=2005, num_steps=total_steps)

    positions = {county: normalize_coordinates(lat, lon) for county, (lat, lon) in county_coordinates.items()}

    step_size = max(1, total_steps // 10)
    time_slider_marks = {i: time_labels[i] for i in range(0, total_steps, step_size) if i < total_steps}

    app.layout = html.Div([ 
        html.H1("Hungary Chickenpox Cases - Temporal GNN Visualization", style={'textAlign': 'center'}),

        html.Div([
            html.Div(
                dl.Map(
                    [
                        dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"),
                        dl.LayerGroup(id='markers')
                    ],
                    center=[47.1625, 19.5033],
                    zoom=7,
                    style={'width': '100%', 'height': '500px'}, 
                    id='map'
                ),
                style={'width': '50%', 'display': 'inline-block'}
            ),

            html.Div(
                cyto.Cytoscape(
                    id='cytoscape-graph',
                    elements=[],
                    layout={'name': 'preset'},
                    style={'width': '100%', 'height': '500px'},
                    stylesheet=[
                        {
                            'selector': 'node',
                            'style': {
                                'label': 'data(label)',
                                'background-color': 'data(color)',
                                'font-size': '14px',
                                'text-wrap': 'wrap',
                                'width': '40px',
                                'height': '40px',
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
                                'width': 1
                            }
                        }
                    ]
                ),
                style={'width': '50%', 'display': 'inline-block'}
            )
        ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}),

        dcc.Slider(
            id='time-slider',
            min=0, max=total_steps - 1, step=1, value=0,
            marks=time_slider_marks
        ),

        html.Div(id='info-output', style={'marginTop': 20, 'fontSize': '16px'}),
        
        # Add comparison graph
        html.Div([
            html.H3("County Case Comparison - Actual vs Predicted", style={'textAlign': 'center'}),
            dcc.Graph(id='comparison-graph')
        ], style={'marginTop': 30}),
        
        # Display scaling info
        html.Div([
            html.H3("Scaling Factors", style={'textAlign': 'center'}),
            html.Div(id='scaling-info')
        ], style={'marginTop': 20})
    ])

    @app.callback(
        [Output('cytoscape-graph', 'elements'),
         Output('markers', 'children'),
         Output('comparison-graph', 'figure'),
         Output('scaling-info', 'children')],
        Input('time-slider', 'value')
    )
    def update_visualization(time_index):
        snapshot = snapshots[time_index]
        
        # Get the original data for this time step from the raw dataset
        time_date = time_labels[time_index].split(' ')
        month, year = time_date[0], int(time_date[1])
        
        # Filter raw data for this time step
        raw_data_filtered = raw_data[
            (raw_data['Date'].dt.month == datetime.datetime.strptime(month, "%B").month) & 
            (raw_data['Date'].dt.year == year)
        ]
        
        # If we have data for this time step, use it
        if not raw_data_filtered.empty:
            true_cases = raw_data_filtered[csv_counties].values[0]
        else:
            # Use a default or placeholder value
            true_cases = np.zeros(len(csv_counties))
            
        # Get predicted cases for this time step (using raw scale predictions)
        if time_index < len(raw_scale_predictions):
            predicted_cases = raw_scale_predictions[time_index]
        else:
            predicted_cases = np.zeros(len(csv_counties))
        
        # Ensure predictions are integers
        predicted_cases = np.round(predicted_cases).astype(int)
        predicted_cases = np.maximum(predicted_cases, 0)  # Ensure no negative values
        
        # Create color mapping based on case counts
        max_cases = max(np.max(true_cases), 1)  # Prevent division by zero
        
        # Generate colors based on intensity (red for higher cases)
        def get_color(cases):
            intensity = min(cases / max_cases, 1.0)
            r = int(255 * intensity)
            g = int(100 * (1 - intensity))
            b = int(100 * (1 - intensity))
            return f'rgb({r},{g},{b})'
        
        # Create network graph nodes
        nodes = []
        markers = []
        
        for i, county in enumerate(csv_counties):
            # Get true case count from raw data
            true_case_count = int(true_cases[i])
            predicted_case_count = int(predicted_cases[i])
            color = get_color(true_case_count)
            
            # Create cytoscape node
            nodes.append({
                'data': {
                    'id': county,
                    'label': f'{county}\n{true_case_count}\nPred: {predicted_case_count}',
                    'color': color,
                    'true_cases': true_case_count,
                    'predicted_cases': predicted_case_count
                },
                'position': positions[county]
            })
            
            # Create map marker
            lat, lon = county_coordinates[county]
            markers.append(
                dl.Marker(
                    position=[lat, lon],
                    children=[
                        dl.Tooltip(f"{county}: {true_case_count} cases (Pred: {predicted_case_count})")
                    ]
                )
            )
        
        # Create edges based on geographical proximity
        edges = []
        for i, county1 in enumerate(csv_counties):
            for j, county2 in enumerate(csv_counties):
                if i < j:  # Avoid duplicate edges
                    lat1, lon1 = county_coordinates[county1]
                    lat2, lon2 = county_coordinates[county2]
                    # Calculate distance
                    distance = ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5
                    # Connect nodes if they're close enough
                    if distance < 1.0:  # Approximately 100km
                        edges.append({
                            'data': {
                                'source': county1,
                                'target': county2
                            }
                        })
        
        # Create comparison graph
        comparison_figure = {
            'data': [
                {
                    'x': csv_counties,
                    'y': true_cases,
                    'type': 'bar',
                    'name': 'Actual Cases',
                    'marker': {'color': 'rgb(255, 50, 50)'}
                },
                {
                    'x': csv_counties,
                    'y': predicted_cases,
                    'type': 'bar',
                    'name': 'Predicted Cases',
                    'marker': {'color': 'rgb(50, 50, 255)'}
                }
            ],
            'layout': {
                'title': f'County Cases for {time_labels[time_index]}',
                'xaxis': {'title': 'County', 'tickangle': 45},
                'yaxis': {'title': 'Number of Cases'},
                'barmode': 'group',
                'legend': {'x': 0, 'y': 1.1, 'orientation': 'h'},
                'margin': {'l': 50, 'r': 50, 'b': 100, 't': 50}
            }
        }
        
        # Create scaling info table
        scaling_info = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("County"), 
                    html.Th("Scaling Factor"), 
                    html.Th("Offset")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(county),
                    html.Td(f"{scaling_factors[i]:.2f}"),
                    html.Td(f"{scaling_offsets[i]:.2f}")
                ]) for i, county in enumerate(csv_counties)
            ])
        ], style={'width': '100%', 'border': '1px solid black', 'borderCollapse': 'collapse'})
        
        return nodes + edges, markers, comparison_figure, scaling_info

    @app.callback(
        Output('info-output', 'children'),
        [Input('cytoscape-graph', 'tapNodeData'),
         Input('time-slider', 'value')]
    )
    def display_click_info(node_data, time_index):
        current_time = time_labels.get(time_index, "Unknown")

        if node_data:
            county = node_data['id']
            true_cases = node_data['true_cases']
            predicted_cases = node_data['predicted_cases']
            error = abs(true_cases - predicted_cases)
            error_percent = (error / (true_cases + 1)) * 100  # Adding 1 to avoid division by zero
            
            return (f"Current time: {current_time} | County: {county} | "
                    f"Actual cases: {true_cases} | Predicted cases: {predicted_cases} | "
                    f"Error: {error:.1f} cases ({error_percent:.1f}%)")

        return f"Current time: {current_time} | Click on a county to see case details."

    return app

# Convert time steps to real dates
def generate_time_labels(start_year=2004, num_steps=500):
    start_date = datetime.date(start_year, 1, 1)
    dates = [start_date + datetime.timedelta(weeks=i) for i in range(num_steps)]
    return {i: date.strftime('%B %Y') for i, date in enumerate(dates)}

# Main Function
def main():
    # Load the dataset
    raw_data = load_raw_data()
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    
    # Create the dashboard with dataset snapshots and scaling
    app = create_dashboard(dataset, raw_data, csv_counties)
    app.run_server(debug=True)

if __name__ == "__main__":
    main()