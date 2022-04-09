import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx
import random


class DAGSampler:
	'''
	Object for sampling DAGs and ADMGs, as well as for sampling probabilities over edges.
	:param num_nodes: number of <observed> nodes (int)
	:param admg: whether to sample ADMGs with a random number of unobserved confounders (boolean)
	'''
	def __init__(self, library=None, num_nodes=3, admg=False):

		assert isinstance(num_nodes, int), 'num_nodes should be an integer'
		assert num_nodes > 0, 'num_nodes should be a positive integer'
		assert isinstance(admg, bool), 'admg should be boolean'


		if library == None:
			self.library = []
		else:
			self.library = library

		self.num_nodes = num_nodes
		self.admg = admg  # whether to also sample bidirected edges
		# calculate the number of elements in a lower triangular matrix num_nodes x num_nodes
		# this is also the number of possible pairs
		self.num_pairs = num_nodes * (num_nodes - 1) / 2

	def _dag_gen(self, num_nodes):
		'''
		:param num_nodes: number of desired observed nodes in graph (int)
		:return: graph (nx.MultiDiGraph objet)
		'''
		# randomly generate edges for graph
		ran_lotril = np.random.randint(0, high=2, size=int(self.num_pairs))

		# populate the lower triangular matrix with the randomly generated edges
		idx = np.tril_indices(num_nodes, k=-1)
		matrix = np.zeros((num_nodes, num_nodes)).astype(int)
		matrix[idx] = ran_lotril
		return nx.from_numpy_matrix(matrix, create_using=nx.MultiDiGraph)

	def _generate_canonical_library(self, num_nodes, graph_history, verbose=False):
		'''
		:param num_nodes: number of desired observed nodes in graph (int)
		:param graph_history: a list of networkx MultiDiGraph objects which have previously been sampled (list[nx.MultiDiGraph])
		:param verbose: whether to print out confirmation of isomorphic checks (boolean)
		:return: graph_nx: canonical graph (nx.MultiDiGraph object) if found, else None
		:return: already_sampled: flag for whether most recently sampled graph already exists in the graph history / library
		'''
		# generate a networkx object digraph (DAG)
		graph_nx = self._dag_gen(num_nodes=num_nodes)

		if self.admg:
			graph_nx = self._add_unobserved(graph_nx)

		ug_nx = self._dag_to_ug(graph_nx)  # transform sampled DAG into UG

		already_sampled = False
		for d_nx in graph_history:
			ug_d_nx = self._dag_to_ug(d_nx)  # transform history into UG
			already_sampled = nx.is_isomorphic(ug_d_nx, ug_nx)  # check if isomorphic with any in history

			if already_sampled == False:
				pass
			else:
				break

		if not already_sampled:  # if we haven't already sampled this canonical form, return it
			if verbose == True:
				print('New graph found.')
			return graph_nx, already_sampled
		else:
			if verbose == True:
				print('Graph already sampled.')
			return None, already_sampled

	def _dag_to_ug(self, d_graph):
		'''
	    Turns a directed graph into an undirected graph to facilitate isomorphic test
	    :param d_graph: directed graph (networkx MultiDiGraph object)
	    :return UG: undirected graph (networkx MultiDiGraph object)
	    '''
		num_nodes = d_graph.number_of_nodes()
		UG = nx.Graph()
		a = num_nodes
		for edge in d_graph.edges:
			from_ = edge[0]
			to_ = edge[1]
			UG.add_edge(from_, a)
			UG.add_edge(a, to_)
			UG.add_edge(from_, a + 1)
			UG.add_edge(a + 1, a)
			a += 2
		return UG

	def _edge_probas(self, graph):
		'''
		Randomly samples edge probabilities for each edge in the inputted graph
		:param graph: (MultiDiGraph object)
		:return probas: probabilities for each edge in the graph (always above 0.5)
		'''
		# gets edge probabilities
		num_edges = len(graph.edges)
		probas = np.random.uniform(0.5, 1, size=num_edges)
		return probas

	def _add_unobserved(self, graph):
		'''
		Randomly selects the number of unobserved confounders to add between pairs of observed variables and
		adds them to the graph. e.g. 1  2  becomes A <- U1 -> B
		:param graph: directed graph with only <observed> variables (nx.MultiDiGraph object)
		:return graph: directed graph with both observed <and> unobserved variables (nx.MultiDiGraph object)
		'''
		# pick number of unobserved
		num_unobs = np.random.randint(1, self.num_nodes)  # at least one unobserved var for admg
		# list all nodes
		nodes = np.arange(self.num_nodes)
		# enumerate all possible pairs
		pairs = [comb for comb in combinations(nodes, 2)]
		# pick num_obs from list of possible pairs
		unobs_links = [pairs[i] for i in range(num_unobs)]

		for i, nodes in enumerate(unobs_links):
			from_ = nodes[0]
			to_ = nodes[1]
			u_name = 'U{}'.format(i)
			graph.add_node(u_name)
			graph.add_edge(u_name, from_, weight=1)
			graph.add_edge(u_name, to_, weight=1)

		return graph

	def generate_library(self, plot=False, verbose=False, max_iters=100, epsilon=0.1):
		'''
		For a given number of maximum iterations, find the canonical graphs for a set of nodes. If self.admg == True
		then this will also find the set of canonical graphs including unobserved confounders. Note that epislon is
		used as a threshold for graph discovery rate. As the number of identified unique graphs increases, discovery rate
		decreases. Therefore, the loop will break when this rate falls below epsilon.
		:param plot: whether to plot the discovered graphs (boolean)
		:param verbose: whether to print isomorphic check output (boolean)
		:param max_iters: maximum number of graph searches (int)
		:param epsilon: minimum discovery rate threshold
		:return self.library: library of canonical graphs
		'''

		assert isinstance(max_iters, int), 'max_iters should be a positive integer'
		assert isinstance(plot, bool), 'plot should be boolean'
		assert isinstance(verbose, bool), 'verbose should be boolean'
		assert isinstance(epsilon, float), 'epsilon should be a positive float between 0 and 1'
		assert (epsilon >= 0) and (epsilon <= 1), 'epsilon should be between 0 and 1'
		assert max_iters > 0, 'max_iters should be a positive integer'


		t = trange(max_iters, desc='Discovery Rate:', leave=True)
		new_graphs = 0

		for i in t:

			graph, already_sampled_flag = self._generate_canonical_library(num_nodes=self.num_nodes,
			                                                               graph_history=self.library,
			                                                               verbose=verbose)

			if (graph != None) and (plot == True):
				self.show_graph(graph, directed=True)

			if already_sampled_flag:
				pass
			else:
				new_graphs += 1
				self.library.append(graph)

			q = new_graphs / (i + 1)  # new graph discovery rate
			t.set_description("Discovery Rate: {})".format(np.round(q, 4)))
			t.refresh()  # to show immediately the update

			if q <= epsilon:
				print('Graph discovery rate fallen below epsilon = {}'.format(epsilon))
				break
		return self.library

	def edge_weighting(self, graph):
		'''
		Adds weights to edges. If admg then it also ensures the two edges from each unobserved confounder have the same weight.
		:param graph: graph for which to assign edge probabilities (nx.MultiDiGraph object)
		:return graph:  graph with assigned edge probabilities (nx.MultiDiGraph object)
		'''
		edge_weights = self._edge_probas(graph)

		for i, e in enumerate(graph.edges(data=True)):
			from_ = e[0]
			to_ = e[1]
			graph[from_][to_][0]['weight'] = edge_weights[i]

		if self.admg:
			U_nodes = [node for node in graph.nodes() if 'U' in str(node)]
			for node in U_nodes:
				edge = list(graph.edges(node, data=True))
				from_ = edge[0][1]
				to_ = edge[1][1]
				edge_w1 = edge[0][2]['weight']
				edge_w2 = edge[1][2]['weight']
				av_weight = (edge_w1 + edge_w2) / 2
				graph[node][from_][0]['weight'] = av_weight
				graph[node][to_][0]['weight'] = av_weight

		return graph

	def show_graph(self, graph, directed, weights=False):
		# function for plotting a graph (directed or undirected)
		labels = nx.get_edge_attributes(graph, 'weight')
		pos = nx.spring_layout(graph)
		nx.draw(graph, pos, node_size=500, with_labels=True, arrows=directed, connectionstyle='arc3, rad = 0.1')
		if weights:
			labels = dict([((u, v,), f"{d['weight']:.2f}") for u, v, d in graph.edges(data=True)])
			nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
		plt.show()


if __name__ == "__main__":
	# A1. Example usage for DAGs (no unobserved confounders)
	num_nodes = 3
	admg = False
	epsilon = 0.1  # minimum graph discovery rate

	# A2. Initialise DAGSampling object:
	ds = DAGSampler(library=None, num_nodes=num_nodes, admg=admg)
	# A3. generate canonical library:
	library = ds.generate_library(plot=False, verbose=False, max_iters=200, epsilon=0.1)
	print('Discovered {} unique DAGs.'.format(len(library)))

	# A4. Sample from library
	graph = random.choice(library)
	# A5. Assign edge probabilities
	proba_graph = ds.edge_weighting(graph=graph)
	# A6. Show graph with randomly assigned edge probabilities
	ds.show_graph(proba_graph, directed=True, weights=True)
	# A7. Get graph info
	edge_info = proba_graph.edges(data=True)
	print(edge_info)


	# B1. Example usage for ADMGS (includes unobserved confounders)
	num_nodes = 3
	admg = True
	epsilon = 0.1  # minimum graph discovery rate

	# B2. Initialise DAGSampling object:
	ds = DAGSampler(library=None, num_nodes=num_nodes, admg=admg)
	# B3. generate canonical library of ADMGs (including unobserved confounders:
	library = ds.generate_library(plot=False, verbose=False, max_iters=200, epsilon=0.1)
	print('Discovered {} unique ADMGs.'.format(len(library)))

	# B4. Sample from library
	graph = random.choice(library)
	# B5. Assign edge probabilities
	proba_graph = ds.edge_weighting(graph=graph)
	# B6. Show graph with randomly assigned edge probabilities
	ds.show_graph(proba_graph, directed=True, weights=True)
	# B7. Get graph info
	edge_info = proba_graph.edges(data=True)
	print(edge_info)

