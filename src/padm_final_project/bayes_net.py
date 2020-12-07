"""Module containing the BayesNet class."""
import networkx as nx
import graphviz as gv


class BayesNet:
    """Representation of a Bayesian Network.

    Attributes:
    nodes (dict{str->Node}): dict mapping each nodes name to its node object.
    graph (DiGraph): network x visual representation of the bayes net.
    """

    nodes = None
    graph = None

    def __init__(self, nodes, graph):
        """
        Create the Bayesian Network.

        Parameters:
            nodes (dict{name(str) -> node(Node)}): dict mapping each node's name to its node object.
        """
        self.nodes = nodes
        self.graph = graph

        
    def get_MAP_estimate(self, q_nodes, observations=dict()):
        """
        Compute the most likely configuration of the network given the evidence.

        Parameters:
            o_nodes (dict(str->value)): dict mapping node name to its observed value
        """
        raise NotImplementedError()

        
    def get_posterior_1(self, q_nodes, o_nodes=dict()):
        """
        Compute the posterior distribution for the query nodes given observations.
        
        USING CHAIN RULE

        Parameters:
            q_nodes (list[string]): list of string names of nodes queried nodes
            o_nodes (dict): dictionary of observed string node names with respective True/False observations
        Return:
            dictionary of most likely probabilities
        """
        posteriors = dict()
        # for each node in the query, call the get posterior of a single node
        reuse_posteriors = dict()
        for node_str in q_nodes:
            (posteriors[node_str], reuse_posteriors) = self.get_posterior_of_node_1(node_str, o_nodes, reuse_posteriors)
        return posteriors
        

    def get_posterior_of_node_1(self, node_str, o_nodes=dict(), reuse_posteriors=dict()):
        """
        Compute the posterior distribution for the query nodes given observations.
        
        USING CHAIN RULE

        Parameters:
            node_str (string): node name
            o_nodes (dict): dictionary of observed string node names with respective True/False observations
            reuse_posteriors (dict): dictionary of previously calculated posteriors; 
                                    although this doesn't matter for get_posterior_of_node_1(), 
                                    it is relevant for get_posterior_1()

        Return:
            a tuple:
                probability of the node evaluating to True given the obsevations
                a reuse_posteriors dictionary
        """

        node = self.nodes[node_str]

        # check if the posterior of this node has previously been evaluated already
        if node_str in reuse_posteriors:
            return (reuse_posteriors[node_str], reuse_posteriors)

        # check if the node has been observed
        if node_str in o_nodes:
            return (1,reuse_posteriors) if o_nodes[node_str] else (0,reuse_posteriors)

        # check if the node has no parents
        if len(node.parents) == 0:
            # this node has not parent yet it wasn't observed - return its posterior
            node_posterior = node.probabilities["prob"][0]
            reuse_posteriors[node_str] = node_posterior
            return (node_posterior, reuse_posteriors)

        # our node has parents
        else:
            # get posterior porbabilities for each parent
            parents_posteriors = dict()
            for parent in node.parents:
                (parents_posteriors[parent.name], reuse_posteriors) = self.get_posterior_of_node_1(parent.name, o_nodes, reuse_posteriors)
            # let's calculate posterior probability of this node
            node_posterior = 0.0
            # go through the conditional probability table; 
            # for each row, ~multiply every value in the row (instead of true/flase use posterior probability for true false)
            for i in range(len(node.probabilities)):
                temp = node.probabilities["prob"][i]
                # for each row, ~multiply every value in the row (instead of true/flase use posterior probability for true false)
                for parent_name in parents_posteriors:
                    temp *= parents_posteriors[parent_name] if node.probabilities[parent_name][i] else (1-parents_posteriors[parent_name])
                # add these values up
                node_posterior += temp
            reuse_posteriors[node_str] = node_posterior
            return (node_posterior, reuse_posteriors)


    def bucket_elimination(self, q_nodes, observations):
        """
        Perform bucket elimination. 

        Parameters:
            q_nodes: a list of string names for query nodes
            observations: a dictionary mapping node names to observations
        """
        # 1. (initialize buckets)
        ordering = get_be_ordering(self, q_nodes, observations) # list
        cpts = [nodes[node].probabiliteis for node in ordering]
        buckets = dict()
        for b in ordering: 
            buckets[b] = []
        # Go through each cpt function, add it to the highest ranked bucket for which the cpt includes the bucket's node
        for cpt in reversed(cpts):
            for node in reversed(ordering):
                if node in cpt.columns:
                    buckets[node].append(cpt)
                    break


        # 2. (Backwards)
        for i, node in reversed(list(enumerate(ordering[1:]))):
            i += 1
            print('node', node, 'i', i)
            print('ordering', ordering[:i])
            # If an observation exists for the current bucket
            if node in observations.keys():
                for j, func in enumerate(buckets[node]):
                    buckets[node].pop(j)
                    new_func = func[func[node] == observations[node]]
                    new_func = new_func.drop(columns=[node])
                    new_node = get_new_node(new_func, buckets, ordering)
                    buckets[new_node].append(new_func)

            else:
                max_or_sum = max if (node in q_nodes) else sum
                product_func = bucket_product(buckets[node])
                eliminated_func = eliminate_node(node, product_func, max_or_sum)
                new_node = get_new_node(eliminated_func, buckets, ordering)
                buckets[new_node].append(eliminated_func)
                if max_or_sum == max:
                    argmax_func = eliminate_node(node, product_func, np.argmax)
                    argmax_func.rename(columns={'prob': node}, inplace=True)
                    argmax_func[node] = argmax_func[node].apply(lambda x: x==1)
                    buckets[node] = [argmax_func]

        final_node = ordering[0]
        final_product = bucket_product(buckets[final_node])
        buckets[final_node] = [final_product]

        return buckets

    
    @staticmethod
    def bucket_product(funcs):
    """
    Creates a single function that is the product of multiple functions.
    Used by bucket elimination to get the product of all the functions in a bucket.
    """
    # Get set of nodes involved in all the functions
    nodes = set()
    for func in funcs: 
        nodes = nodes.union(set(func.columns))
    nodes.remove('prob')
    nodes = list(nodes)
    
    # Create all permutations of node assignments
    node_values = [True, False] # Assumes binary nodes (must implement different logic to get node_values if nodes aren't binary)
    perms = set()
    for c in itertools.combinations_with_replacement(node_values, len(nodes)):
        for p in itertools.permutations(c):
            perms.add(p)
            
    # Convert func dfs to dfs with a multiindex corresponding to node assignments
    # Only column left is 'prob' column
    mi_funcs = []
    for func in funcs:
        midx = pd.MultiIndex.from_frame(func[func.columns[:-1]])
        mi_funcs.append(func[['prob']].set_index(midx))
        
    # Compute the probability for each assignment and create new df
    rows = []
    for perm in perms:
        assignment = dict(zip(nodes, perm))
        product = 1
        for mi_func in mi_funcs:
            key = tuple(assignment[node] for node in mi_func.index.names)
            product *= mi_func.loc[key]['prob']
        rows.append(list(perm)+[product])
    return pd.DataFrame(rows, columns=nodes+['prob'])

    @staticmethod
    def eliminate_node(node, func, elim_func):
    
    """
    Takes a function and a node to sum out of the function.
    Used by bucket elimination at each bucket after func_product() to sum out the bucket's node.
    """
    node_values = [True, False]
    nodes = list(func.columns)
    nodes.remove(node)
    nodes.remove('prob')
    
    # Get all permutations of node assignmnets not being summed out
    perms = []
    for perm in func[nodes].drop_duplicates().iterrows():
        perms.append(tuple(perm[1]))
    print('perms', perms)
    midx = pd.MultiIndex.from_frame(func[nodes])
    mi_func = func[[node, 'prob']].set_index(midx)
    
    rows = []
    for perm in perms:
        probability = elim_func(mi_func.loc[perm].prob)
        rows.append(list(perm)+[probability])
    return pd.DataFrame(rows, columns=nodes+['prob'])


    @staticmethod
    def get_new_node(func, buckets, ordering):
        """
        Helper function for bucket elimination. Returns the bucket in which to place the provided function.
        """
        for node in reversed(ordering):
            if node in func.columns:
                return node
    
        
    def get_order(self, nodes, f):
        """Greedy Search for constructing bucket elimination ordering"""
        nodes = set(self.nodes.keys()) - q_nodes.union(observations.keys())
        graph = self.graph.subgraph(nodes).copy()
        hidden_ordering = []
        for _ in range(len(nodes)):
            node = argmin_of_f(f, nodes, graph)
            hidden_ordering.append(node)
            nodes.remove(node)
            graph.remove_node(node)
        return list(q_nodes) + hidden_ordering + list(observations.keys())


    @staticmethod
    def argmin_of_f(f, nodes, graph):
        """Returns the argmin (over nodes) of f(node, graph) """
        min_node = None
        min_score = np.infty
        for node in nodes:
            score = f(node, graph)
            if score < min_score:
                min_score = score
                min_node = node
        return node

    @staticmethod
    def order_heuristic(node, graph):
        """Min-neighbors heuristic: The cost of a node is the number of neighbors it has in the current graph"""
        return len(graph.succ[node]) + len(graph.pred[node])

    def draw_net(self, ax=None):
        """Draw a diagram of the network."""
        nx.draw(
            self.graph,
            with_labels=True,
            pos=nx.get_node_attributes(self.graph, "pos"),
            node_color="orange",
            node_size=500,
            ax=ax,
        )

    def get_cpt_diagram(self, rank_labels=["a", "s", "w"], format_type=None, size=None):
        """
        Draw the graph (but have CPTs as labels).

        Args:
            rank_labels (List[str]): Labels to place on the same rank in the graph

        Returns:
            bytes: Rendered PNG of graph
        """
        dot_graph = gv.Digraph(strict=True)
        if format_type is not None:
            dot_graph.format = format_type
        if size is not None:
            dot_graph.size = size

        key_to_subgraph = {
            key: gv.Digraph(name=key, graph_attr={"rank": "same"})
            for key in rank_labels
        }

        for node_key, node in self.nodes.items():
            if node_key[0] in key_to_subgraph:
                graph_to_use = key_to_subgraph[node_key[0]]
            else:
                graph_to_use = dot_graph

            graph_to_use.node(node_key, node.get_html_cpt(), shape="oval")

        for _, subgraph in key_to_subgraph.items():
            dot_graph.subgraph(subgraph)

        for edge in self.graph.edges:
            dot_graph.edge(*edge)

        return dot_graph
