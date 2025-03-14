import networkx as nx
import matplotlib.pyplot as plt

# Function to draw and save a graph
def draw_graph(G, title, filename, pos=None):
    plt.figure(figsize=(6, 4))
    if not pos:
        pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', arrowstyle='-|>', arrowsize=20)
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

# Undirected Graph
G_undirected = nx.Graph()
G_undirected.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
draw_graph(G_undirected, 'Undirected Graph', 'undirected_graph.png')

# Directed Graph
G_directed = nx.DiGraph()
G_directed.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
draw_graph(G_directed, 'Directed Graph', 'directed_graph.png')

print("Basic graph visuals saved as 'undirected_graph.png' and 'directed_graph.png'.")
