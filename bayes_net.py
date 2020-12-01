# Draft of Bayes Net class
class Bayes_Net():
    """ Representation of a Baysian Network.

    Attributes:
    nodes (dict{str->Node}): dict mapping each nodes name to its node object.
    graph (DiGraph): network x visual representation of the bayes net.
    """

    nodes = None
    graph = None

    def __init__(self, nodes, graph):
        """ Creates the Bayesian Network

        Parameters:
        nodes (dict{name(str) -> node(Node)}): dict mapping each node's name to its node object.
        """
        self.nodes = nodes
        self.graph = graph



    def add_observation(self, node, observation):
        """ Adds an observation of a node to the network.

        Parameters:
        node (Node): node that was observed
        observation ( ): value of the observation
        """
        raise NotImplementedError()


    def remove_observation(self, node):
        """ Removes an observation from the network.

        Parameters:
        node (Node): node thats observation is being removed
        """
        raise NotImplementedError()


    def add_node(self, Node):
        """ Adds a node to the nework. Must be a leaf node.

        Parameters:
        node (Node): node to be added to the network
        """
        raise NotImplementedError()


    def remove_node(self, node):
        """ Remove a node from the network. Must be a leaf node (it would affect child nodes otherwise).

        Parameters:
        node (Node): leaf node to be removed from the network
        """
        raise NotImplementedError()


    def get_MAP_estimate(self, o_nodes=[]):
        """ Compute the most likely configuration of the network given the evidence.
        o_nodes (list(Node)): List of observed nodes (Node value =\= None)
        """

        raise NotImplementedError()


    def get_posterior(self, q_nodes, o_nodes=[]):
        """Compute the posterior distribution for the query nodes (q_nodes)
            given the observed nodes (o_nodes).

        Parameters:
        q_nodes (: Set of nodes for which the posterior is computed
        o_nodes (set(Node)): Set of observed nodes.
        """
        raise NotImplementedError()


    def bucket_elimination(self, o_nodes, attack_nodes):
       """
        Knowing which variables are observations and which are the results (attacks),
        perform bucket elimination on the bayes net.
        This simplifies the computation in the case where hidden nodes reamin unobservable,
        and observations are asummed to occur for a given set of nodes.
        This allows for faster subsequent computation.

        Parameters:
        o_nodes: a list of string names for observation nodes
        attach_nodes: a list of string names for attack nodes


       """


    def draw_net(self):
        """Draws a diagram of the network."""
        nx.draw(self.graph, with_labels=True, pos = nx.get_node_attributes(self.graph,'pos'), node_color = 'orange', node_size = 500)
