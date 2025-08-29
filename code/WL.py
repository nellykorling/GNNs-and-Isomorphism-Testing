
import networkx as nx
import hashlib
import matplotlib.pyplot as plt
import os
from torch_geometric.datasets import TUDataset as tud
from torch_geometric.utils import to_networkx
from collections import Counter
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def stable_hash(label):
    '''
    stable hash function
    '''
    return hashlib.sha256(str(label).encode('utf-8')).hexdigest()

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

def update_node_label(G, node, i):
    '''
    update the node label to be a hash of the node itself and the 
    multiset of its neighbors
    '''
    neighbors = G.neighbors(node)
    neighbor_labels = sorted(str(G.nodes[neighbor]['label']) for neighbor in neighbors)
    label = (str(G.nodes[node]['label']),) + tuple(neighbor_labels)
    G.nodes[node]['label'] = str(i) + stable_hash(label)


def WL(G, iterations):
    '''
    repeat node label updates for a given number of iterations
    and return the multiset of all nodes' labels from all iterations
    '''
    initialize_node_labels(G)
    graph_label = []
    for i in range(1,iterations+1):
        for node in G.nodes():
            update_node_label(G, node, i)
        graph_label.extend([G.nodes[node]['label'] for node in G.nodes()])
    return Counter(graph_label)


def WL_test_isomorphic(G1, G2, iterations):
    '''
    test if two graphs are isomorphic by comparing the output of the WL algorithm
    for the graphs. If the multisets are equal, the graphs may be isomorphic.
    '''
    return WL(G1, iterations) == WL(G2, iterations)


def WL_kernel_entry(G1_multiset, G2_multiset):
    '''
    compute the kernel matrix for two graphs
    '''
    G1_alphabet = set(G1_multiset.keys())
    G2_alphabet = set(G2_multiset.keys())

    #make union of alphabets of G1 and G2
    alphabet = sorted(G1_alphabet.union(G2_alphabet))

    #count occurrences of each label in the alphabet in G1 and G2 to form feature vectors for G1 and G2
    G1_feature_vector = [G1_multiset[label] for label in alphabet]
    G2_feature_vector = [G2_multiset[label] for label in alphabet]

    #inner product of feature vectors to form kernel enteries
    kernel = np.dot(G1_feature_vector, G2_feature_vector)
    return kernel


def WL_kernel_matrix(dataset):
    '''
    compute the kernel matrix for a dataset of graphs
    '''
    kernel_matrix = np.zeros((len(dataset), len(dataset)))
    for i in range(len(dataset)):
        for j in range(i, len(dataset)):
            kernel_matrix[i, j] = WL_kernel_entry(WL(to_networkx(dataset[i]), 3), WL(to_networkx(dataset[j]), 3))
            kernel_matrix[j, i] = kernel_matrix[i, j]
    return kernel_matrix


def svm_with_kernel(K, y, C=1.0, n_splits=10, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_accs = [] # training accuarcies at each fold
    test_accs = []

    for train_idx, test_idx in skf.split(K, y):
        K_train = K[np.ix_(train_idx, train_idx)]

        # kernel-based methods need comparison of test to train to compute value for test
        K_test = K[np.ix_(test_idx, train_idx)] 
        y_train, y_test = y[train_idx], y[test_idx]

        clf = SVC(kernel='precomputed', C=C)
        clf.fit(K_train, y_train)

        # training accuracy
        y_train_pred = clf.predict(K_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_accs.append(train_acc)

        # testing accuracy
        y_test_pred = clf.predict(K_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_accs.append(test_acc)
    
    print("average training accuracy:", np.mean(train_accs))
    print("average testing accuracy:", np.mean(test_accs))
    return train_accs, test_accs



# MUTAG dataset
dataset = tud(root='./data', name='MUTAG')

mutag_kernel_matrix= WL_kernel_matrix(dataset)

svm_with_kernel(mutag_kernel_matrix, dataset.y)