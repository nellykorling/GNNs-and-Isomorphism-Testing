

from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset
import numpy as np


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')


indeces = np.arange(len(dataset))
print(indeces)

# other update node labels for WL.py

    # neighbors = G.neighbors(node)
    # label = []
    # for neighbor in neighbors:
    #     label.append(str(G.nodes[neighbor]['label']))
    # label = sorted(label)
    # label.insert(0, str(G.nodes[node]['label'])) 
    # G.nodes[node]['label'] = str(i) + stable_hash(label)


    # multiset = [str(G.nodes[neighbor]['label']) for neighbor in neighbors]
    # label = (str(G.nodes[node]['label']),) + tuple(sorted(multiset))
    # G.nodes[node]['label'] = str(i) + stable_hash(label)