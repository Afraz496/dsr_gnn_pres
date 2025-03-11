import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Load chickenpox case data
data = pd.read_csv('hungary_chickenpox.csv')
data = data.melt(id_vars=['Date'], var_name='county', value_name='cases')
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data = data.sort_values('Date')

# Load edge list for counties
edges = pd.read_csv('hungary_county_edges.csv')

# Create the graph
G = nx.from_pandas_edgelist(edges, 'name_1', 'name_2')

# Assign positions for consistent plotting
pos = nx.spring_layout(G, seed=42)

# Get unique dates for animation
time_steps = sorted(data['Date'].unique())

# Global normalization for all cases
min_cases, max_cases = data['cases'].min(), data['cases'].max()

# Function to normalize case counts
def normalize_cases(cases):
    return (cases - min_cases) / (max_cases - min_cases)

# Function to update the graph for each time step
def update(num):
    plt.clf()
    date = pd.to_datetime(time_steps[num])
    week_data = data[data['Date'] == date].set_index('county')['cases']

    # Assign colors and sizes based on case counts
    case_values = np.array([week_data.get(node, 0) for node in G.nodes()])
    colors = plt.cm.Reds(normalize_cases(case_values))
    sizes = 300 + (case_values / max_cases * 1000)  # Adjust size scaling

    nodes = nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.Reds, node_size=sizes)
    edges = nx.draw_networkx_edges(G, pos, edge_color='gray')
    labels = nx.draw_networkx_labels(G, pos)
    plt.title(f'Chickenpox Cases - {date.strftime("%d-%m-%Y")}')

    # Add colorbar for reference
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min_cases, vmax=max_cases))
    sm.set_array([])
    plt.colorbar(sm, label='Number of Cases')

    return nodes, edges, labels

# Create animation
fig = plt.figure(figsize=(10, 8))
ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=1000, blit=False, repeat=True)

# To save the animation, uncomment the next line
# ani.save('chickenpox_spread.gif', writer='pillow')

plt.show()
