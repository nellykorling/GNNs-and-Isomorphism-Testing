import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from torch_geometric.datasets import TUDataset
from sklearn.metrics import accuracy_score
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
import random
from sklearn.model_selection import StratifiedKFold



def initialize_node_features(G, method='degree'):
    features = []

    if method == "degree":
        for node in G.nodes():
            features.append([G.degree[node]])
    elif method == "constant":
        for node in G.nodes():
            features.append([1.0])

    return torch.tensor(features, dtype=torch.float)


def get_adjacency_list(G):
    return [list(G.neighbors(node)) for node in G.nodes()]


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

def graph_pooling(X, method):
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


class GNN_Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.gnn = MPGNN(in_dim, hidden_dim, num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, G):
        X = initialize_node_features(G)  # degree-based init
        neighbors = get_adjacency_list(G)
        node_embeds = self.gnn(X, neighbors)
        graph_embed = graph_pooling(node_embeds, method='sum')  # [hidden_dim]
        return self.classifier(graph_embed)  # [num_classes]

# Load MUTAG dataset
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')



def gnn_with_cv(dataset, n_splits=5, epochs=50, lr=0.01, hidden_dim=32, num_layers=2, seed=42):

    indices = np.arange(len(dataset))
    num_classes = dataset.num_classes
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_accs, test_accs = [], []

    for train_idx, test_idx in skf.split(indices, dataset.y):
        model = GNN_Classifier(in_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for i in train_idx:
                data = dataset[i]
                G = to_networkx(data)
                label = data.y
                out = model(G)                               # [num_classes]
                loss = loss_fn(out.unsqueeze(0), label)      # cross entropy requires batch dim, [1, num_classes] vs [1]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        
        model.eval()
        # train accuracy
        y_true_tr, y_pred_tr = [], []
        with torch.no_grad():
            for i in train_idx:
                data = dataset[i]
                G = to_networkx(data)
                y_true_tr.append(data.y.item())
                y_pred_tr.append(model(G).argmax().item())
        train_acc = accuracy_score(y_true_tr, y_pred_tr)
        train_accs.append(train_acc)

        # test accuracy
        y_true_te, y_pred_te = [], []
        with torch.no_grad():
            for i in test_idx:
                data = dataset[i]
                G = to_networkx(data)
                y_true_te.append(data.y.item())
                y_pred_te.append(model(G).argmax().item())
        test_acc = accuracy_score(y_true_te, y_pred_te)
        test_accs.append(test_acc)

    print("Average training accuracy:", np.mean(train_accs))
    print("Average testing accuracy:",  np.mean(test_accs))
    return train_accs, test_accs


#train_accs, test_accs = gnn_with_cv(dataset)

