import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
import requests
from urllib.request import urlopen

# Load the results
data = np.load('model_results.npz')
true_values = data['true_values']
catboost_predictions = data['catboost_predictions']
mpnn_predictions = data['mpnn_predictions']
node_indices = data['node_indices']
timestamps = data['timestamps']

# Since get_areas doesn't exist, we'll use node indices directly
# We'll create pseudonyms for the areas
unique_nodes = np.unique(node_indices)
node_to_area = {node: f"Area_{node}" for node in unique_nodes}
area_names = [f"Area_{node}" for node in unique_nodes]

# Prepare data for visualization
df = pd.DataFrame({
    'true_values': true_values,
    'catboost_predictions': catboost_predictions,
    'mpnn_predictions': mpnn_predictions,
    'node_index': node_indices,
    'timestamp': timestamps,
    'area_name': [node_to_area[node] for node in node_indices]
})


# Create a pivot table for time series visualization
pivot_df = df.pivot_table(
    index=['timestamp'], 
    columns=['area_name'], 
    values=['true_values', 'catboost_predictions', 'mpnn_predictions']
)

# Calculate metrics
catboost_mse = mean_squared_error(true_values, catboost_predictions)
catboost_mae = mean_absolute_error(true_values, catboost_predictions)
catboost_r2 = r2_score(true_values, catboost_predictions)

mpnn_mse = mean_squared_error(true_values, mpnn_predictions)
mpnn_mae = mean_absolute_error(true_values, mpnn_predictions)
mpnn_r2 = r2_score(true_values, mpnn_predictions)

# Create Dash app
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server
app.title = "COVID-19 Prediction Dashboard"

# Define custom CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Create app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("COVID-19 Spread and Prediction Dashboard - England", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'margin-bottom': '20px'}),
        html.P("Comparing MPNN-LSTM and CatBoost models for COVID-19 case prediction",
               style={'textAlign': 'center', 'color': '#7f8c8d'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin-bottom': '20px'}),
    
    # Model comparison metrics
    html.Div([
        html.Div([
            html.H3("Model Performance Metrics", style={'textAlign': 'center', 'color': '#2c3e50'}),
            html.Div([
                html.Div([
                    html.H4("MPNN-LSTM", style={'textAlign': 'center', 'color': '#c0392b'}),
                    html.P(f"MSE: {mpnn_mse:.4f}", style={'textAlign': 'center'}),
                    html.P(f"MAE: {mpnn_mae:.4f}", style={'textAlign': 'center'}),
                    html.P(f"R²: {mpnn_r2:.4f}", style={'textAlign': 'center'})
                ], style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '15px', 'margin': '10px', 'backgroundColor': '#f8f9fa'}),
                html.Div([
                    html.H4("CatBoost", style={'textAlign': 'center', 'color': '#27ae60'}),
                    html.P(f"MSE: {catboost_mse:.4f}", style={'textAlign': 'center'}),
                    html.P(f"MAE: {catboost_mae:.4f}", style={'textAlign': 'center'}),
                    html.P(f"R²: {catboost_r2:.4f}", style={'textAlign': 'center'})
                ], style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '15px', 'margin': '10px', 'backgroundColor': '#f8f9fa'})
            ], style={'display': 'flex', 'justifyContent': 'center'})
        ], className="six columns"),
        
        html.Div([
            html.H3("Time Series Visualization", style={'textAlign': 'center', 'color': '#2c3e50'}),
            dcc.Dropdown(
                id="select-area",
                options=[{"label": area, "value": area} for area in area_names[:10]],  # Top 10 areas
                value=area_names[0],
                style={'marginBottom': '10px'}
            ),
            dcc.Graph(id="time-series-graph")
        ], className="six columns")
    ], className="row", style={'marginBottom': '20px'}),
    
    # Prediction comparison
    html.Div([
        html.H3("Prediction Comparison", style={'textAlign': 'center', 'color': '#2c3e50'}),
        dcc.Dropdown(
            id="comparison-timestamp",
            options=[{"label": f"Timestamp {int(t)}", "value": t} for t in np.unique(timestamps)],
            value=np.unique(timestamps)[0],
            style={'marginBottom': '10px'}
        ),
        dcc.Graph(id="prediction-comparison")
    ], className="row", style={'marginBottom': '20px'}),
    
    # Error Analysis
    html.Div([
        html.H3("Error Analysis", style={'textAlign': 'center', 'color': '#2c3e50'}),
        dcc.Graph(id="error-distribution")
    ], className="row", style={'marginBottom': '20px'}),
    
    # Top Areas Performance
    html.Div([
        html.H3("Performance for Top Areas", style={'textAlign': 'center', 'color': '#2c3e50'}),
        dcc.Graph(id="top-areas-performance")
    ], className="row", style={'marginBottom': '20px'}),
    
    # Footer
    html.Div([
        html.P("MPNN-LSTM and CatBoost Model Comparison for COVID-19 Prediction",
               style={'textAlign': 'center', 'color': '#7f8c8d'}),
        html.P("Based on England COVID-19 Data from PyTorch Geometric Temporal",
               style={'textAlign': 'center', 'color': '#7f8c8d'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'marginTop': '20px'})
], className="container", style={'fontFamily': 'Arial', 'maxWidth': '1200px', 'margin': 'auto'})

# Callbacks
@app.callback(
    Output("time-series-graph", "figure"),
    [Input("select-area", "value")]
)
def update_time_series(selected_area):
    # Filter data for the selected area
    area_data = df[df['area_name'] == selected_area]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=area_data['timestamp'], 
        y=area_data['true_values'],
        mode='lines+markers',
        name='Actual Cases',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=area_data['timestamp'], 
        y=area_data['mpnn_predictions'],
        mode='lines+markers',
        name='MPNN-LSTM Predictions',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=area_data['timestamp'], 
        y=area_data['catboost_predictions'],
        mode='lines+markers',
        name='CatBoost Predictions',
        line=dict(color='green', dash='dot', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=f'COVID-19 Cases in {selected_area}',
        xaxis_title='Time',
        yaxis_title='Case Count (Normalized)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified"
    )
    
    return fig

@app.callback(
    Output("prediction-comparison", "figure"),
    [Input("comparison-timestamp", "value")]
)
def update_prediction_comparison(selected_timestamp):
    # Filter data for the selected timestamp
    timestamp_data = df[df['timestamp'] == selected_timestamp]
    
    # Create scatter plot
    fig = make_subplots(rows=1, cols=2, subplot_titles=["MPNN-LSTM vs Actual", "CatBoost vs Actual"])
    
    # MPNN-LSTM vs Actual
    fig.add_trace(
        go.Scatter(
            x=timestamp_data['true_values'],
            y=timestamp_data['mpnn_predictions'],
            mode='markers',
            name='MPNN-LSTM',
            marker=dict(color='red', size=8)
        ),
        row=1, col=1
    )
    
    # Add perfect prediction line
    max_val = max(timestamp_data['true_values'].max(), timestamp_data['mpnn_predictions'].max()) * 1.1
    min_val = min(timestamp_data['true_values'].min(), timestamp_data['mpnn_predictions'].min()) * 0.9
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='black', dash='dash')
        ),
        row=1, col=1
    )
    
    # CatBoost vs Actual
    fig.add_trace(
        go.Scatter(
            x=timestamp_data['true_values'],
            y=timestamp_data['catboost_predictions'],
            mode='markers',
            name='CatBoost',
            marker=dict(color='green', size=8)
        ),
        row=1, col=2
    )
    
    # Add perfect prediction line
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            showlegend=False,
            line=dict(color='black', dash='dash')
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Model Predictions vs Actual Values at Timestamp {int(selected_timestamp)}",
        height=500,
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        xaxis2_title="Actual Values",
        yaxis2_title="Predicted Values"
    )
    
    return fig

@app.callback(
    Output("error-distribution", "figure"),
    [Input("comparison-timestamp", "value")]
)
def update_error_distribution(selected_timestamp):
    # Filter data for the selected timestamp
    timestamp_data = df[df['timestamp'] == selected_timestamp]
    
    # Calculate errors
    mpnn_errors = timestamp_data['mpnn_predictions'] - timestamp_data['true_values']
    catboost_errors = timestamp_data['catboost_predictions'] - timestamp_data['true_values']
    
    # Create figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Error Distribution", "Error by Area"])
    
    # Error histogram
    fig.add_trace(
        go.Histogram(
            x=mpnn_errors,
            name="MPNN-LSTM Errors",
            opacity=0.7,
            marker_color='red'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=catboost_errors,
            name="CatBoost Errors",
            opacity=0.7,
            marker_color='green'
        ),
        row=1, col=1
    )
    
    # Area-wise errors for top 10 areas with highest cases
    top_areas = timestamp_data.sort_values(by='true_values', ascending=False).head(10)['area_name'].unique()
    top_areas_data = timestamp_data[timestamp_data['area_name'].isin(top_areas)]
    
    fig.add_trace(
        go.Bar(
            x=top_areas_data['area_name'],
            y=abs(top_areas_data['mpnn_predictions'] - top_areas_data['true_values']),
            name="MPNN-LSTM |Error|",
            marker_color='red'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=top_areas_data['area_name'],
            y=abs(top_areas_data['catboost_predictions'] - top_areas_data['true_values']),
            name="CatBoost |Error|",
            marker_color='green'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Error Analysis at Timestamp {int(selected_timestamp)}",
        height=500,
        barmode='group',
        xaxis_title="Error",
        yaxis_title="Count",
        xaxis2_title="Area",
        yaxis2_title="Absolute Error"
    )
    
    fig.update_xaxes(tickangle=45, row=1, col=2)
    
    return fig

@app.callback(
    Output("top-areas-performance", "figure"),
    [Input("comparison-timestamp", "value")]
)
def update_top_areas_performance(selected_timestamp):
    # Calculate average error by area across all timestamps
    area_avg_error = df.groupby('area_name').apply(
        lambda x: pd.Series({
            'mpnn_mae': mean_absolute_error(x['true_values'], x['mpnn_predictions']),
            'catboost_mae': mean_absolute_error(x['true_values'], x['catboost_predictions']),
            'avg_cases': x['true_values'].mean()
        })
    ).reset_index()
    
    # Get top 10 areas by case count
    # Get top 10 areas by case count
    top_areas = area_avg_error.sort_values(by='avg_cases', ascending=False).head(10)
    
    # Create figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=["MPNN-LSTM MAE by Area", "CatBoost MAE by Area"])

    # MPNN-LSTM MAE by area
    fig.add_trace(
        go.Bar(
            x=top_areas['area_name'],
            y=top_areas['mpnn_mae'],
            name="MPNN-LSTM MAE",
            marker_color='red'
        ),
        row=1, col=1
    )
    
    # CatBoost MAE by area
    fig.add_trace(
        go.Bar(
            x=top_areas['area_name'],
            y=top_areas['catboost_mae'],
            name="CatBoost MAE",
            marker_color='green'
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title="Top Areas Performance Comparison",
        height=500,
        barmode='group',
        xaxis_title="Area",
        yaxis_title="Mean Absolute Error",
        xaxis2_title="Area",
        yaxis2_title="Mean Absolute Error"
    )
    
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)