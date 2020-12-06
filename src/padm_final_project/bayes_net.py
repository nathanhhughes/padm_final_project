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

    def get_MAP_estimate(self, o_nodes=[]):
        """
        Compute the most likely configuration of the network given the evidence.

        Parameters:
            o_nodes (list(Node)): List of observed nodes
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


    def bucket_elimination(self, o_nodes, attack_nodes):
        """
        Perform bucket elimination to solve for node ordering.

        Knowing which variables are observations and which are the results
        (attacks), perform bucket elimination on the bayes net.  This
        simplifies the computation in the case where hidden nodes remain
        unobservable, and observations are assumed to occur for a given set of
        nodes.  This allows for faster subsequent computation.

        Parameters:
            o_nodes: a list of string names for observation nodes
            attack_nodes: a list of string names for attack nodes
        """
        raise NotImplementedError()

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
