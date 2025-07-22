import torch
import torch.nn as nn
import networkx as nx
import numpy as np


G = nx.cycle_graph(6)


def initialize_node_features(G, method='degree'):
    features = []

    if method == "degree":
        for node in G.nodes():
            features.append([G.degree[node]])
    elif method == "constant":
        for node in G.nodes():
            features.append("1")

    return torch.tensor(features, dtype=torch.float)

print(initialize_node_features(G).size(0))

def get_adjacency_list(G):
    return [list(G.neighbors(node)) for node in G.nodes()]

print(len(get_adjacency_list(G)))

class GNN_Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        '''
        in_dim: input feature dimension of nodes (int)
        out_dim: desired output feature dimension of 
                    node embeddings (int)
        '''
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias = False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias = False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.activation = nn.ReLU()

    def forward(self, X, neighbors):
        '''
        X: node feature matrix (number of nodes, in_dim)
        neighbors: adjacency list for each node

        returns: 
        h_new: output feature matrix (number of nodes, out_dim)
        '''
        h_new = torch.zeros(X.size(0), self.W_self.out_features)
        for u in range(X.size(0)):
            h_self = self.W_self(X[u]) 

            h_neigh = torch.zeros_like(h_self)
            for v in neighbors[u]:
                h_neigh += self.W_neigh(X[v])  

            h_new[u] = h_self + h_neigh + self.bias
        return self.activation(h_new)

class MPGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        """
        in_dim: feature dimension of input nodes
        hidden_dim: output dimension of each GNN layer
        num_layers: number of GNN layers to apply
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GNN_Layer(in_dim, hidden_dim)) 

        for _ in range(num_layers - 1):
            self.layers.append(GNN_Layer(hidden_dim, hidden_dim)) 

    def forward(self, X, neighbors):
        """
        x: [num_nodes, input_dim] node feature matrix
        adj_list: adjacency list

        applies each layer in sequence

        Returns:
        x: [num_nodes, hidden_dim] final node embeddings
        """
        for layer in self.layers:
            X = layer(X, neighbors)
        return X

def graph_pooling(X, method='mean'):
    """
    pool node embeddings into graph embedding

    method: 'sum' or 'mean'
    X: node feature matrix (num_nodes, dim)

    Returns:
    graph_embedding: (out_dim)
    """
    if method == 'sum':
        return X.sum(dim=0)
    elif method == 'mean':
        return X.mean(dim=0)
    else:
        raise ValueError("Unknown pooling method: choose 'sum' or 'mean'.")


G = nx.cycle_graph(6)

#create same graph with permuted nodes
G_perm = nx.cycle_graph(6)
G_perm = nx.relabel_nodes(G_perm, {i: i + 1 for i in range(6)})


X = initialize_node_features(G)       
neighbors = get_adjacency_list(G) 

model = MPGNN(in_dim=1, hidden_dim=8, num_layers=2)
node_embeddings = model(X, neighbors)
graph_embedding = graph_pooling(node_embeddings, method='sum')

node_embeddings_perm = model(X, neighbors)
graph_embedding_perm = graph_pooling(node_embeddings_perm, method='sum')

print("Graph embedding:", graph_embedding)
print("Graph embedding (permuted):", graph_embedding_perm)