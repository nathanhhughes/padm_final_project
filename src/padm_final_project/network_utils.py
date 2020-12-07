"""Module containing helpful functionality for generating examples."""
from .node import Node
from .bayes_net import BayesNet
import networkx as nx
import pandas as pd
import numpy as np
import random


def generate_network(n_attacks=2, n_subsystems=3, n_workstations=6, seed=None):
    """
    Make an example network with the number of components.

    Args:
      n_attacks (int): number of "attack" random variables
      n_subsystems (int): number of "subsystem" random variables
      n_workstations (int): number of "workstation" random variables

    Returns:
      BayesNet: a bayes net with the specified random variables.

    """
    if seed is not None:
        np.random.seed(seed)

    # TODO(someone) think about moving the graph building to the BayesNet class

    G = nx.DiGraph()
    graph_width = max(n_attacks, n_subsystems, n_workstations)

    # handle inadequate inputs
    if n_attacks == 0:
        raise ValueError("Need at least one attack!")
    if n_workstations == 0:
        raise ValueError("Need at least one workstation!")

    # generate workstation nodes;
    workstations_nodes = []
    for i in range(n_workstations):
        G.add_node(
            "w" + str(i), pos=(graph_width * (i + 0.5) / n_workstations, 4), color="r"
        )
        workstations_nodes.append(Node("w" + str(i), generate_random_cpt([])))

    # generate subsystem nodes
    subsystem_nodes = []
    for i in range(n_subsystems):
        # a random sample of 1 through n workstations
        G.add_node(
            "s" + str(i), pos=(graph_width * (i + 0.5) / n_subsystems, 2), color="b"
        )
        sample_workstations = random.sample(
            range(0, n_workstations), random.randint(1, n_workstations)
        )
        for j in sample_workstations:
            G.add_edge("w" + str(j), "s" + str(i))
        subsystem_nodes.append(
            Node(
                "s" + str(i),
                generate_random_cpt(["w" + str(j) for j in sample_workstations]),
                parents=[workstations_nodes[j] for j in sample_workstations],
            )
        )

    # generate attack nodes;
    attacks_nodes = []
    for i in range(n_attacks):
        G.add_node(
            "a" + str(i), pos=(graph_width * (i + 0.5) / n_attacks, 0), color=""
        )

        if n_subsystems != 0:
            sample_workstations = random.sample(
                range(0, n_workstations), random.randint(0, n_workstations // 2)
            )  # sample some workstations
            sample_subsystems = random.sample(
                range(0, n_subsystems), random.randint(1, n_subsystems)
            )  # sample at least one subsystem
            for j in sample_workstations:
                G.add_edge("w" + str(j), "a" + str(i))
            for j in sample_subsystems:
                G.add_edge("s" + str(j), "a" + str(i))
            subsystem_nodes.append(
                Node(
                    "a" + str(i),
                    generate_random_cpt(
                        ["w" + str(j) for j in sample_workstations]
                        + ["s" + str(j) for j in sample_subsystems]
                    ),
                    [workstations_nodes[j] for j in sample_workstations]
                    + [subsystem_nodes[j] for j in sample_subsystems],
                )
            )
        else:
            # there are no subsystems, sample just attacks
            sample_workstations = random.sample(
                range(0, n_workstations), random.randint(1, n_workstations)
            )
            for j in sample_workstations:
                G.add_edge("w" + str(j), "a" + str(i))
            attacks_nodes.append(
                Node(
                    "a" + str(i),
                    generate_random_cpt(["w" + str(j) for j in sample_workstations]),
                    [workstations_nodes[j] for j in sample_workstations],
                )
            )

    dictionary_of_nodes = dict()
    all_nodes = attacks_nodes + subsystem_nodes + workstations_nodes
    for n in all_nodes:
        dictionary_of_nodes[n.name] = n
    return BayesNet(dictionary_of_nodes, G)


def generate_random_cpt(parents=[]):
    """
    Make a random cpt from from a given list of parents.

    Assumes that the value of nodes are either True or False.

    Args:
      parents (list of str): list of parents; can be empty

    Returns:
      pd.DataFrame: a cpt in a dataframe representation;

    """
    df = pd.DataFrame(columns=parents + ["prob"])
    ar = [False] * len(parents)
    # we need to create 2^parents rows - the cpt table.
    # these few lines generate all combinations of false / true statements
    for i in range(2 ** len(parents)):
        df.loc[i] = ar + [random.random()]
        j = 0
        while j < len(parents) and True:
            if not ar[j]:
                ar[j] = True
                ar[0:j] = [False] * (j)
                break
            else:
                j += 1

    return df
