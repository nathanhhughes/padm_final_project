"""Module containing the BayesNet class."""
import networkx as nx
import graphviz as gv
import pandas as pd
import numpy as np
import itertools
import random


class BayesNet:
    """Representation of a Bayesian Network.

    Attributes:
    nodes (dict{str->Node}): dict mapping each nodes name to its node object.
    graph (DiGraph): network x visual representation of the bayes net.
    """

    nodes = None
    graph = None

    def __init__(self, nodes, graph=None):
        """
        Create the Bayesian Network.

        Parameters:
            nodes (dict{name(str) -> node(Node)}): dict mapping each node's name to its node object.
        """
        self.nodes = {node.name: node for node in nodes}
        if graph is not None:
            self.graph = graph
        else:
            self.graph = nx.DiGraph()
            for node in nodes:
                self.graph.add_node(node.name, color="")
                for parent in node.parents:
                    self.graph.add_edge(parent.name, node.name)

    def get_random_workstation_observations(self, seed = None):
        """
        Create a dictionary of random workstation observations

        Returns:
            o_nodes (dict(str->bool)): dictionary of observed string node names with respective True/False observations
        """
        if seed is not None:
            random.seed(seed)

        o_nodes = dict()
        for node_str in self.nodes:
            # check to see if that's a workstation node
            if node_str[0] == 'w':
                o_nodes[node_str] = random.random() > 0.5
        return o_nodes


    def get_MAP_estimate(self, q_nodes, observations):
        """
        Compute the most likely configuration of the network given the evidence.

        Parameters:
            q_nodes (set(str)): set of nodes to get MAP estimate for
            observations (dict(str->value)): dict mapping node name to its observed value
        """
        ordering = self.get_order(q_nodes, observations)
        q_nodes_set = set(q_nodes)
        buckets = self.bucket_elimination(q_nodes, observations, do_map=True)
        ## Prints/displays information for debugging purposes (uncomment if needed to debug)
        # print(ordering)
        # print("--Post BE--")
        # for node in ordering:
        #     print(f'--bucket {node.upper()}--')
        #     if buckets[node] is None:
        #         print("*Empty*")
        #     else:
        #         print(buckets[node])
        #         for func in buckets[node]:
        #             display(func)
        initial_node = ordering[0]
        argmax_idx = np.argmax(buckets[ordering[0]]["prob"])
        map_estimate = {initial_node: buckets[initial_node][initial_node].iloc[argmax_idx]}
        for node in ordering[1:]:
            if node not in q_nodes_set:
                return map_estimate
            else:
                argmax_node = buckets[node]
                if isinstance(argmax_node.index, pd.MultiIndex):
                    assignment = tuple(map_estimate[node] for node in argmax_node.index.names if node in map_estimate.keys())
                    map_estimate[node] = argmax_node.loc[assignment][node]
                else:
                    map_estimate[node] = argmax_node[node][0]
        return map_estimate


    def get_posterior(self, q_nodes, observations):
        """
        Compute the posterior distribution for the query nodes given observations.

        Parameters:
            q_nodes (list[string]): list of string names of nodes queried nodes
            observations (dict(str->bool)): dictionary of observed string node names with respective True/False observations
        Return:
            dictionary of most likely probabilities
        """
        ordering = self.get_order(q_nodes, observations)
        funcs = self.bucket_elimination(q_nodes, observations)
        posterior = BayesNet.bucket_product(funcs)
        
        # Normalize (bucket elimination returns unnormalized posterior)
        alpha = sum(posterior.prob)
        posterior.prob = posterior.prob.apply(lambda x: x/alpha)
        
        return posterior


    def bucket_elimination(self, q_nodes, observations, do_map=False):
        """
        Perform bucket elimination.

        Parameters:
            q_nodes: a list of string names for query nodes
            observations: a dictionary mapping node names to observations
        """

        # 1. (Initialize buckets step)
        ordering = BayesNet.get_order(self, q_nodes, observations)
        cpts = [BayesNet.get_cpt_for_bucket(node, self.nodes[node].probabilities) for node in ordering]
        buckets = dict()
        for b in ordering: 
            buckets[b] = []
        # Go through each cpt function, add it to the highest ranked bucket for which the cpt includes the bucket's node
        for cpt in reversed(cpts):
            for node in reversed(ordering):
                if node in cpt.columns:
                    buckets[node].append(cpt)
                    break
                    
        ## Prints/displays information for debugging purposes (uncomment if needed to debug)
        # print('--initial buckets--')
        # for node in ordering:
        #     print(f"--Bucket {node.upper()}--")
        #     for func in buckets[node]:
        #         display(func)
        ##
        
        # 2. (Backwards step)
        q_nodes_set = set(q_nodes)
        for i, node in reversed(list(enumerate(ordering))):

            ## Prints/displays information for debugging purposes (uncomment if needed to debug)
            # print(f'--bucket {node.upper()}--')
            # if len(buckets[node]) == 0:
            #     print("*Empty*")
            # for func in buckets[node]:
            #     display(func)
            ##

            if i == 0:
                break

            # If an observation exists for the current bucket
            if node in observations.keys():
                for func in buckets[node]:
                    new_func = func[func[node] == observations[node]]
                    new_func = new_func.drop(columns=[node])
                    new_node = BayesNet.get_new_node(new_func, buckets, ordering[:i])
                    buckets[new_node].append(new_func)
                buckets[node]=None # Empty bucket after all funcs are moved out of it

            elif node in q_nodes_set and not do_map: 
                break

            else:
                max_or_sum = max if (node in q_nodes) else sum
                product_func = BayesNet.bucket_product(buckets[node])
                eliminated_func = BayesNet.eliminate_node(node, product_func, max_or_sum)
                new_node = BayesNet.get_new_node(eliminated_func, buckets, ordering[:i])
                buckets[new_node].append(eliminated_func)

                if max_or_sum == max and do_map:
                    argmax_func = BayesNet.eliminate_node(node, product_func, np.argmax)
                    if len(argmax_func.columns) > 1:
                        midx = pd.MultiIndex.from_frame(argmax_func[argmax_func.columns[:-1]])
                        argmax_func = argmax_func[[node]].set_index(midx)
                    buckets[node] = argmax_func
        
        if not do_map:
            remaining_funcs = []
            for node in q_nodes:
                if buckets[node]:
                    remaining_funcs.extend(buckets[node])

            return remaining_funcs

        else:
            final_node = ordering[0]
            final_product = BayesNet.bucket_product(buckets[final_node])
            buckets[final_node] = final_product

            return buckets


    @staticmethod
    def get_cpt_for_bucket(node, cpt):
        """Converts the cpt format used by Node.probabilities into the format used by BayesNet.bucket_elimination()"""
        expanded_cpts = []
        for value in (True, False):
            cpt_copy = cpt.copy(deep=True)
            cpt_copy.insert(0, node, value)
            if not value:
                cpt_copy.prob = cpt_copy.prob.apply(lambda x: 1-x)
            expanded_cpts.append(cpt_copy)
        return pd.concat(expanded_cpts, axis=0)


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
        scalars = []
        for func in funcs:
            if len(func.columns) > 1:
                midx = pd.MultiIndex.from_frame(func[func.columns[:-1]])
                mi_funcs.append(func[['prob']].set_index(midx))
            else:
                scalars.append(func)

        # Compute the probability for each assignment and create new df
        rows = []
        for perm in perms:
            assignment = dict(zip(nodes, perm))
            product = 1
            for mi_func in mi_funcs:
                key = tuple(assignment[node] for node in mi_func.index.names)
                product *= mi_func.loc[key]['prob']
            rows.append(list(perm)+[product])
        new_func = pd.DataFrame(rows, columns=nodes+['prob'])
        for scalar in scalars:
            new_func.prob = new_func.prob.apply(lambda x: scalar.prob*x)
        return new_func


    @staticmethod
    def eliminate_node(node, func, elim_func):

        """
        Takes a function and a node to sum out of the function.
        Used by bucket elimination at each bucket after func_product() to sum out the bucket's node.
        """
        use_argmax = elim_func == np.argmax
        
        node_values = [True, False]
        nodes = list(func.columns)
        nodes.remove(node)
        nodes.remove('prob')

        if nodes:
            # Get all permutations of node assignmnets not being summed out
            perms = []
            for perm in func[nodes].drop_duplicates().iterrows():
                perms.append(tuple(perm[1]))

            midx = pd.MultiIndex.from_frame(func[nodes])
            mi_func = func[[node, 'prob']].set_index(midx)
        else:
            if not use_argmax:
                result = pd.DataFrame([elim_func(func.prob)], columns=['prob'])
            else:
                idx = elim_func(func.prob)
                result = pd.DataFrame([func[node].iloc[idx]], columns=[node])
            return result

        rows = []
        for perm in perms:
            if not use_argmax:
                value = elim_func(mi_func.loc[perm].prob)
            else:
                idx = elim_func(mi_func.loc[perm].prob)
                value = mi_func.loc[perm][node].iloc[idx]
            rows.append(list(perm)+[value])
        new_func = pd.DataFrame(rows, columns=nodes+[node if use_argmax else 'prob'])
        
        ## Prints/displays information for debugging purposes (uncomment if needed to debug)
        # if not use_argmax:
        #     print("bucket product:")
        #     display(func)
        #     print(f"eliminated {node} with {elim_func.__name__}. New func:")
        #     display(new_func)
        ##

        return new_func

    @staticmethod
    def get_new_node(func, buckets, ordering):
        """
        Helper function for bucket elimination. Returns the bucket in which to place the provided function.
        """
        for node in reversed(ordering):
            if node in func.columns:
                return node
        return ordering[-1]

    def get_order(self, q_nodes, observations):
        """Greedy Search for constructing bucket elimination ordering"""
        nodes = set(self.nodes.keys()) - set(q_nodes).union(set(observations.keys()))
        graph = self.graph.subgraph(nodes).copy()
        hidden_ordering = []
        for _ in range(len(nodes)):
            node = BayesNet.argmin_of_f(nodes, graph)
            hidden_ordering.append(node)
            nodes.remove(node)
            graph.remove_node(node)
        return list(q_nodes) + hidden_ordering + list(observations.keys())


    @staticmethod
    def argmin_of_f(nodes, graph):
        """Returns the argmin (over nodes) of f(node, graph) """
        min_node = None
        min_score = np.infty
        for node in nodes:
            score = len(graph.succ[node]) + len(graph.pred[node])
            if score < min_score:
                min_score = score
                min_node = node
        return node

    def draw_net(self, ax=None):
        """Draw a diagram of the network."""
        try:
            nx.draw(
                self.graph,
                with_labels=True,
                pos=nx.get_node_attributes(self.graph, "pos"),
                node_color="orange",
                node_size=500,
                ax=ax,
            )
        except:
            nx.draw(
                self.graph,
                with_labels=True,
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
