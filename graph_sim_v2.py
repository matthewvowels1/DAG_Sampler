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
    return nx.from_numpy_matrix(matrix, create_using=nx.DiGraph)


def dag_to_ug(dag):
    num_nodes = dag.number_of_nodes()
    UG = nx.Graph()
    a = num_nodes
    for edge in dag.edges:
        from_ = edge[0]
        to_ = edge[1]
        UG.add_edge(from_, a)
        UG.add_edge(a, to_)
        UG.add_edge(from_, a + 1)
        UG.add_edge(a + 1, a)
        a += 2
    return UG

def generate_canonical_library(num_nodes, graph_history):

    # generate a networkx object digraph (DAG)
    dag_nx = dag_gen(num_nodes=num_nodes)

    ug_nx = dag_to_ug(dag_nx)  # transform sampled DAG into UG

    already_sampled = False
    for d_nx in graph_history:
        ug_d_nx = dag_to_ug((d_nx))  # transform history into UG
        already_sampled = nx.is_isomorphic(ug_d_nx, ug_nx)  # check if isomorphic with any in history

        if already_sampled == False:
            pass
        else:
            break


    if not already_sampled:  # if we haven't already sampled this canonical form, return it
        print('New graph found.')
        return dag_nx, already_sampled
    else:
        print('Graph already sampled.')
        return None, already_sampled
