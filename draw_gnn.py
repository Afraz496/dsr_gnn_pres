import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import io
from PIL import Image

# Set random seed for reproducibility
np.random.seed(42)

# Create a simple graph structure
def create_graph(num_nodes=7):
    G = nx.gnp_random_graph(num_nodes, 0.4)
    pos = nx.spring_layout(G, seed=42)
    return G, pos

# Initialize node features and hidden states
# Initialize node features and hidden states
def init_features(G, feature_dim=3, hidden_dim=4):
    # Initial node features (random for demonstration)
    node_features = {
        node: np.pad(np.random.rand(feature_dim), (0, hidden_dim - feature_dim), 'constant')
        for node in G.nodes()
    }
    
    # Initial hidden states (zeros)
    hidden_states = {node: np.zeros(hidden_dim) for node in G.nodes()}
    
    return node_features, hidden_states


# Simulate message passing in a GNN layer
def message_passing(G, node_features, hidden_states, time_step):
    # New hidden states after message passing
    new_hidden = {}
    
    for node in G.nodes():
        # Collect messages from neighbors
        neighbor_msgs = []
        for neighbor in G.neighbors(node):
            # Message is a function of neighbor's features and previous hidden state
            msg = 0.6 * node_features[neighbor] + 0.4 * hidden_states[neighbor]
            neighbor_msgs.append(msg)
        
        if neighbor_msgs:
            # Aggregate messages (mean)
            agg_msg = np.mean(neighbor_msgs, axis=0)
            
            # Update hidden state with temporal component
            # New hidden = update(current features, aggregated messages, previous hidden)
            temporal_factor = 0.3 if time_step > 0 else 0
            new_hidden[node] = 0.4 * node_features[node] + 0.3 * agg_msg + temporal_factor * hidden_states[node]
            
            # Apply non-linearity (tanh for demonstration)
            new_hidden[node] = np.tanh(new_hidden[node])
        else:
            # For isolated nodes
            new_hidden[node] = 0.7 * node_features[node]
            if time_step > 0:
                new_hidden[node] += 0.3 * hidden_states[node]
            new_hidden[node] = np.tanh(new_hidden[node])
    
    return new_hidden

# Predict output from hidden states
def predict_output(hidden_states, output_dim=2):
    outputs = {}
    for node, hidden in hidden_states.items():
        # Simple linear transformation for demonstration
        outputs[node] = np.tanh(np.sum(hidden) * np.ones(output_dim) / hidden.shape[0])
    return outputs

# Create custom colormaps
def create_colormaps():
    feature_cmap = LinearSegmentedColormap.from_list('feature_cmap', ['#f7fbff', '#08306b'])
    hidden_cmap = LinearSegmentedColormap.from_list('hidden_cmap', ['#fff5eb', '#7f2704'])
    output_cmap = LinearSegmentedColormap.from_list('output_cmap', ['#f7fcf5', '#00441b'])
    return feature_cmap, hidden_cmap, output_cmap

# Create animation frames
def create_animation(max_time_steps=5):
    # Set up the graph and initial states
    G, pos = create_graph()
    node_features, hidden_states = init_features(G)
    feature_cmap, hidden_cmap, output_cmap = create_colormaps()
    
    # Store frames for animation
    frames = []
    
    # Run TGNN for multiple time steps
    for t in range(max_time_steps):
        # Update hidden states with message passing
        new_hidden = message_passing(G, node_features, hidden_states, t)
        
        # Predict outputs
        outputs = predict_output(new_hidden)
        
        # Create visualization for this time step
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[3, 1])
        
        # Title for the entire figure
        fig.suptitle(f'Temporal Graph Neural Network - Time Step {t}', fontsize=16)
        
        # 1. Graph Structure with Node Features
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_title("Input Graph with Node Features")
        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.5)
        
        feature_values = [np.mean(node_features[node]) for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=feature_values, 
                              cmap=feature_cmap, node_size=500)
        nx.draw_networkx_labels(G, pos, ax=ax1)
        ax1.set_axis_off()
        
        # 2. Message Passing Visualization
        ax2 = plt.subplot(gs[0, 1])
        ax2.set_title("Message Passing")
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=500, node_color='lightgray')
        nx.draw_networkx_labels(G, pos, ax=ax2)
        
        # Draw edges with directional arrows to show message flow
        drawn_edges = set()
        for u, v in G.edges():
            ax2.add_patch(FancyArrowPatch(pos[u], pos[v], 
                                         arrowstyle='->', 
                                         mutation_scale=20, 
                                         color='blue',
                                         alpha=0.6))
            drawn_edges.add((u, v))
            
            # Add reverse direction if not already drawn
            if (v, u) not in drawn_edges:
                ax2.add_patch(FancyArrowPatch(pos[v], pos[u], 
                                             arrowstyle='->',
                                             mutation_scale=20, 
                                             color='blue',
                                             alpha=0.6))
        ax2.set_axis_off()
        
        # 3. Hidden State Visualization
        ax3 = plt.subplot(gs[0, 2])
        ax3.set_title("Hidden States")
        hidden_values = [np.mean(new_hidden[node]) for node in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=ax3, alpha=0.3)
        nx.draw_networkx_nodes(G, pos, ax=ax3, node_color=hidden_values, 
                              cmap=hidden_cmap, node_size=500)
        nx.draw_networkx_labels(G, pos, ax=ax3)
        ax3.set_axis_off()
        
        # 4. Output Visualization
        ax4 = plt.subplot(gs[0, 3])
        ax4.set_title("Predicted Outputs")
        output_values = [np.mean(outputs[node]) for node in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=ax4, alpha=0.3)
        nx.draw_networkx_nodes(G, pos, ax=ax4, node_color=output_values, 
                              cmap=output_cmap, node_size=500)
        nx.draw_networkx_labels(G, pos, ax=ax4)
        ax4.set_axis_off()
        
        # 5. Temporal Information Flow
        ax5 = plt.subplot(gs[1, :])
        ax5.set_title("Temporal Information Flow")
        ax5.set_xlim([-0.5, max_time_steps - 0.5])
        ax5.set_ylim([0, 1])
        ax5.set_xlabel("Time Steps")
        ax5.set_ylabel("Hidden State Magnitude")
        
        # Plot the evolution of hidden states over time
        if t > 0:
            for node in G.nodes():
                # Get historical values from previous frames
                x_vals = list(range(t+1))
                y_vals = [frames[i]['hidden_mean'][node] for i in range(t)] + [np.mean(np.abs(new_hidden[node]))]
                ax5.plot(x_vals, y_vals, marker='o', label=f'Node {node}')
                
        if t == 0:
            # For the first frame, just show the points
            for node in G.nodes():
                ax5.scatter(0, np.mean(np.abs(new_hidden[node])), label=f'Node {node}')
                
        # Add legend for the first and last frames only
        if t == 0 or t == max_time_steps - 1:
            if len(G.nodes()) <= 10:  # Only show legend if not too many nodes
                ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        frames.append({
            'img': img,
            'hidden_mean': {node: np.mean(np.abs(new_hidden[node])) for node in G.nodes()}
        })
        plt.close()
        
        # Update hidden states for next time step
        hidden_states = new_hidden
    
    # Create GIF
    frames[0]['img'].save('tgnn_visualization.gif',
                         format='GIF',
                         append_images=[f['img'] for f in frames[1:]],
                         save_all=True,
                         duration=1000,  # ms per frame
                         loop=0)  # 0 means loop indefinitely
    
    print("Animation saved as 'tgnn_visualization.gif'")
    return frames

# Run the animation
frames = create_animation(max_time_steps=5)