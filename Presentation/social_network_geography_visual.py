import networkx as nx
import matplotlib.pyplot as plt

# Social Network Graph
def create_social_network():
    G = nx.Graph()
    G.add_edges_from([
        ('Alice', 'Bob'),
        ('Alice', 'Charlie'),
        ('Bob', 'David'),
        ('Charlie', 'David'),
        ('David', 'Eve'),
        ('Eve', 'Frank')
    ])

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=12, font_weight='bold')
    plt.title('Social Network Graph')
    plt.savefig('social_network_graph.png')
    plt.close()

# Geography Graph
def create_geography_graph():
    G = nx.Graph()
    G.add_edges_from([
        ('City A', 'City B'),
        ('City A', 'City C'),
        ('City B', 'City D'),
        ('City C', 'City D'),
        ('City D', 'City E')
    ])

    pos = {
        'City A': (0, 0),
        'City B': (1, 1),
        'City C': (1, -1),
        'City D': (2, 0),
        'City E': (3, 0)
    }
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2000, font_size=12, font_weight='bold')
    plt.title('Geographic Graph')
    plt.savefig('geography_graph.png')
    plt.close()

create_social_network()
create_geography_graph()
