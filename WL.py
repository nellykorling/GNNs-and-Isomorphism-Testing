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
# mapping = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}
# G2 = nx.relabel_nodes(G1, mapping)

G2 = nx.Graph()
G2.add_edges_from([(0, 1), (1, 2), (2, 0)])  # First triangle
G2.add_edges_from([(3, 4), (4, 5), (5, 3)])

# Create folder
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

# Draw and save the graphs
plt.figure(figsize=(8, 4))

plt.subplot(121)
nx.draw_shell(G1, with_labels=True)
plt.title("Graph G1: Cycle C6")

plt.subplot(122)
nx.draw_shell(G2, with_labels=True)
plt.title("Graph G2: Two triangles + matching")

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "wl_indistinguishable_graphs.png"))
plt.close()

G3 = nx.path_graph(6)

def stable_hash(label_tuple):
    '''
    stable hash function
    '''
    return hashlib.sha256(str(label_tuple).encode('utf-8')).hexdigest()

def initialize_node_labels(G, method='degree'):
    '''
    initialize node labels as either 1 or the degree of the node
    '''
    for node in G.nodes():
        if method == 'degree':
            G.nodes[node]['label'] = G.degree(node)
        elif method == 'constant':
            G.nodes[node]['label'] = 1
        else:
            raise ValueError(f"Invalid method: {method}")
    return G

def update_node_label(G, node):
    '''
    update the node label to be a hash of the node itself and the 
    multiset of its neighbors
    '''
    neighbors = G.neighbors(node)
    multiset = [str(G.nodes[neighbor]['label'] for neighbor in neighbors)]
    # Include the node's own label in the hash
    label_tuple = (str(G.nodes[node]['label']),) + tuple(sorted(multiset))
    G.nodes[node]['label'] = stable_hash(label_tuple)


def WL(G, iterations):
    '''
    repeat node label updates for a given number of iterations
    and return the multiset of all nodes' labels from all iterations
    '''
    initialize_node_labels(G)
    all_labels = []
    for _ in range(iterations):
        for node in G.nodes():
            update_node_label(G, node)
        # Collect labels after this iteration
        all_labels.append(sorted([G.nodes[node]['label'] for node in G.nodes()]))
    return all_labels


def WL_test_isomorphic(G1, G2, iterations):
    '''
    test if two graphs are isomorphic by comparing the output of the WL algorithm
    for the graphs. If the multisets are equal, the graphs may be isomorphic.
    '''
    multiset1 = WL(G1, iterations)
    multiset2 = WL(G2, iterations)
    return multiset1 == multiset2








