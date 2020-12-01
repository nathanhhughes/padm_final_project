"""Module containing the BayesNet class."""
import networkx as nx


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

    def add_observation(self, node, observation):
        """
        Add an observation of a node to the network.

        Parameters:
            node (Node): node that was observed
            observation ( ): value of the observation
        """
        raise NotImplementedError()

    def remove_observation(self, node):
        """
        Remove an observation from the network.

        Parameters:
            node (Node): node observation that is being removed
        """
        raise NotImplementedError()

    def add_node(self, node):
        """
        Add a node to the network. Must be a leaf node.

        Parameters:
            node (Node): node to be added to the network
        """
        raise NotImplementedError()

    def remove_node(self, node):
        """
        Remove a node from the network.

        The removed node must be a leaf node.

        Parameters:
            node (Node): leaf node to be removed from the network
        """
        raise NotImplementedError()

    def get_MAP_estimate(self, o_nodes=[]):
        """
        Compute the most likely configuration of the network given the evidence.

        Parameters:
            o_nodes (list(Node)): List of observed nodes
        """
        raise NotImplementedError()

    def get_posterior(self, q_nodes, o_nodes=[]):
        """
        Compute the posterior distribution for the query nodes given observations.

        Parameters:
            q_nodes (set(Node)): Set of nodes for which the posterior is computed
            o_nodes (set(Node)): Set of observed nodes.
        """
        raise NotImplementedError()

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
            attach_nodes: a list of string names for attack nodes
        """
        raise NotImplementedError()

    def draw_net(self):
        """Draw a diagram of the network."""
        nx.draw(
            self.graph,
            with_labels=True,
            pos=nx.get_node_attributes(self.graph, "pos"),
            node_color="orange",
            node_size=500,
        )
