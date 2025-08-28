import networkx as nx
import matplotlib.pyplot as plt
import os
from torch_geometric.datasets import TUDataset as tud
from torch_geometric.utils import to_networkx



# Download/load MUTAG dataset
dataset = tud(root='./data', name='MUTAG')

print(f"Number of graphs: {len(dataset)}")
print(dataset[0])

print("Node features:")
print(dataset[0].x)

G1 = to_networkx(dataset[0])
print(G1)

G2 = to_networkx(dataset[1])

#Create folder
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

# Draw and save the graphs
plt.figure(figsize=(8, 4))

plt.subplot(121)
nx.draw_shell(G1, with_labels=True)
plt.title("MUTAG graph 0")

plt.subplot(122)
nx.draw_shell(G2, with_labels=True)
plt.title("MUTAG graph 1")

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "MUTAG_graphs.png"))
plt.close()

'''
# Create the original C6 cycle graph
G1 = nx.cycle_graph(6)

# Create a relabeled isomorphic graph with a permutation of node labels
# Example permutation: 0->3, 1->4, 2->5, 3->0, 4->1, 5->2
# mapping = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}
# G2 = nx.relabel_nodes(G1, mapping)

G2 = nx.Graph()
G2.add_edges_from([(0, 1), (1, 2), (2, 0)]) 
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
'''