import numpy as np

def assign_string(adj_matrix):	# Assign a string to each node
    num_nodes = len(adj_matrix)
    strings_children = []
    for v in range(num_nodes - 1, -1, -1): # Start from the last node and add the information of their children
        child_v = np.nonzero(adj_matrix[:, v])[0]
        if len(child_v) == 0:
            strings_children.append('0')
        else:
            first_child_str = strings_children[num_nodes-1-child_v[0]]
            first_character = str(int(first_child_str[0]) + 1)
            full_str = first_character
            for c in child_v:
                full_str += strings_children[num_nodes-1-c]
            strings_children.append(full_str)
    strings_children.reverse()
    # Now add the information of the parents:
    strings_parent= []
    for v in range(num_nodes):
        full_str = ''
        parent_v = np.nonzero(adj_matrix[v, :])[0]
        for p in parent_v:
            full_str += strings_children[p]
        strings_parent.append(full_str)
    # Append the two strings:
    assigned_strings = []
    for v in range(num_nodes):
        assigned_strings.append(strings_children[v] + 'p' + strings_parent[v])
    return assigned_strings


def unique_dag(adj_mat):
    '''
    We do not want to oversample from certain sets of isomorphic DAGs, we therefore change all incoming graphs into
    a unique graph with the same isomorphic properties.
    :param adj_mat: adjacency matrix sampled as part of a lower triangular matrix
    :return: unique dag which is equivalent to the graph given in adj_mat
    '''
    # Keep only the lower triangle of this matrix to form the adjacency matrix of a DAG:
    adj_matrix = np.tril(adj_mat, k=-1)
    # Get the strings assigned based on the random matrix:
    assigned_strs = assign_string(adj_matrix)
    sorted_idx = list(np.argsort(assigned_strs))
    sorted_idx.reverse()
    matrix = adj_matrix[np.ix_(sorted_idx, sorted_idx)]
    return matrix


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
    canonical_form = unique_dag(matrix)
    return canonical_form


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
    canonical_form = generate_randdag(num_graphs=1, num_nodes=num_nodes)[0]

    # check if canonical graph already exists in graph history and generate a corresponding 'flag'
    already_sampled = any(np.array_equal(canonical_form, g) for g in graph_history)

    if not already_sampled:  # if we haven't already sampled this canonical form, return it
        print('New graph found.')
        return canonical_form, already_sampled
    else:
        print('Graph already sampled.')
        return None, already_sampled
