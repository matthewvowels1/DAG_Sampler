# DAG_Sampler
Samples a DAG with or without unobserved confounders and checks for isomorphisms in previously sampled history.

### Example Usage:

```python
### Parameters ###
from ADMG_Sampler import*
import numpy as np


num_nodes = 5  # number of observed variables
admg = True   # randomly add bidirected edges (in the form of unobserved confounders between pairs of nodes)
seed = 43   # random seed
max_graphs = 200  # will stop searching for non-iso. graphs if we have discovered 200
epsilon = 0.1  # graph discovery rate threshold (will stop searching if discovery rate falls below epsilon)
max_iters = 1000  # will stop searching for graphs if max_iters is exceeded
sparsity = np.log(num_nodes) / num_nodes  # encourages the discovery of sparse graphs
costs = False  # turns edge weight probabilities into costs according to np.log(pe / (1-pe))
rounding = False  # rounds edge weight probabilities to the nearest decimal


# Initialise DAGSampling object:
ds = DAGSampler(library=None, num_nodes=num_nodes, admg=admg, seed=seed)
# generate canonical library of ADMGs (including unobserved confounders:
library = ds.generate_library(plot=False, verbose=False, max_iters=max_iters, epsilon=epsilon,
                              max_graphs=max_graphs, sparsity_param=sparsity_param)
print('Discovered {} ADMGs.'.format(len(library)))

# Sample uniformly from library of NON-isomorphic graphs
graph = random.choice(library)
# Assign edge probabilities / weights
proba_graph = ds.edge_weighting(graph=graph, costs=costs, rounding=rounding)
# Show graph with randomly assigned edge probabilities
ds.show_graph(proba_graph, directed=True, weights=True)
# Get graph info
edge_info = proba_graph.edges(data=True)
print(edge_info)
```

## Acknowledgements:

Thanks to Sina Akbari [BAN, EPFL] for their advice regarding the identification of isomorphisms using a conversion into an undirected graph.