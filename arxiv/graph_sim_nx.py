import numpy as np
import networkx as nx



def dag_gen(num_nodes):
    '''

    :param num_nodes: number of desired nodes in sampled DAG (int)
    :return: canonical form: unique graph for isomorphic set [num_nodes x num_nodes]
    '''
    # calculate the number of elements in a lower triangular matrix num_nodes x num_nodes
    num_elements = num_nodes * (num_nodes - 1) / 2
    # randomly generate edges for graph

    ran_lotril = np.random.randint(0, high=2, size=int(num_elements))

    # populate the lower triangular matrix with the randomly generated edges
    idx = np.tril_indices(num_nodes, k=-1)
    matrix = np.zeros((num_nodes, num_nodes)).astype(int)
    matrix[idx] = ran_lotril
    return matrix


def generate_randdag(num_graphs, num_nodes):
    '''
    Generates the adjacency matrices for a DAG.
    :param num_graphs:  the number of graphs you want to generate
    :param num_nodes:  the number of nodes for each graph
    :return adj_matrices = adjacency matrices for the graphs
    '''

    adj_matrices = np.zeros((num_graphs, num_nodes, num_nodes))

    for i in range(num_graphs):
        # generate graph
        adj_mat = dag_gen(num_nodes)
        adj_matrices[i, :, :] = adj_mat

    return adj_matrices

def generate_canonical_library(num_nodes, graph_history):

    # generate a unique graph (up to some probability)
    graph = generate_randdag(num_graphs=1, num_nodes=num_nodes)[0]

    # check if canonical graph already exists in graph history and generate a corresponding 'flag'
    already_sampled = any(np.array_equal(graph, g) for g in graph_history)

    if not already_sampled:  # if we haven't already sampled this canonical form, return it
        print('New graph found.')
        return graph, already_sampled
    else:
        print('Graph already sampled.')
        return None, already_sampled
