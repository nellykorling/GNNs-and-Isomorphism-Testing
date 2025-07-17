# Weisfeiler-Lehman algorithm for graph isomorphism
# use networkx for graph components
import networkx as nx
import hashlib
import matplotlib.pyplot as plt
import os

# Create the original C6 cycle graph
G1 = nx.cycle_graph(6)

# Create a relabeled isomorphic graph with a permutation of node labels
# Example permutation: 0->3, 1->4, 2->5, 3->0, 4->1, 5->2
mapping = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}
G2 = nx.relabel_nodes(G1, mapping)

G3 = nx.path_graph(6)

# stable hash function
def stable_hash(label_tuple):
    return hashlib.sha256(str(label_tuple).encode('utf-8')).hexdigest()

# initialize node labels as either 1 or the degree of the node
def initialize_node_labels(G, method='degree'):
    for node in G.nodes():
        if method == 'degree':
            G.nodes[node]['label'] = G.degree(node)
        elif method == 'constant':
            G.nodes[node]['label'] = 1
        else:
            raise ValueError(f"Invalid method: {method}")
    return G

# update the node label to be a hash of the node itseld and the multiset of its neighbors
def update_node_label(G, node):
    neighbors = G.neighbors(node)
    multiset = [str(G.nodes[neighbor]['label'] for neighbor in neighbors)]
    # Include the node's own label in the hash
    label_tuple = (str(G.nodes[node]['label']),) + tuple(sorted(multiset))
    G.nodes[node]['label'] = stable_hash(label_tuple)


# repeat for each node for a fixed number of iterations
def WL_iterations(G, iterations):
    for _ in range(iterations):
        for node in G.nodes():
            update_node_label(G, node)
    return G




initialize_node_labels(G1, method='degree')
initialize_node_labels(G2, method='degree')
initialize_node_labels(G3, method='degree')

WL_iterations(G1, 6)
for node in G1.nodes():
    print(node, G1.nodes[node]['label'])
print("--------------------------------")
WL_iterations(G2, 6)
for node in G2.nodes():
    print(node, G2.nodes[node]['label'])
print("--------------------------------")
WL_iterations(G3, 6)
for node in G3.nodes():
    print(node, G3.nodes[node]['label'])




# compare representations of two graphs to determine if they are isomorphic

