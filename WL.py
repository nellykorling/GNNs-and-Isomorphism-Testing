
import networkx as nx
import hashlib
import matplotlib.pyplot as plt
import os
from torch_geometric.datasets import TUDataset as tud
from torch_geometric.utils import to_networkx

from example_graphs import G1, G2

# Download/load MUTAG dataset
dataset = tud(root='./data', name='MUTAG')

G1 = to_networkx(dataset[0])
G2 = to_networkx(dataset[1])

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
        all_labels.extend([G.nodes[node]['label'] for node in G.nodes()])
    return all_labels

def WL_test_isomorphic(G1, G2, iterations):
    '''
    test if two graphs are isomorphic by comparing the output of the WL algorithm
    for the graphs. If the multisets are equal, the graphs may be isomorphic.
    '''
    from collections import Counter
    multiset1 = Counter(WL(G1, iterations))
    print(multiset1)
    multiset2 = Counter(WL(G2, iterations))
    print(multiset2)
    return multiset1 == multiset2



print(WL_test_isomorphic(G1, G2, 4))

def kernel_matrix(G1, G2):
    return


def SVM_classifier(kernel_matrix, C):
    return













