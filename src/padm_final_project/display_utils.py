"""Module containing some quick helper functions for pretty printing an example."""


def display_state(network, observations, add_newline=True):
    """Show observed values of the nodes of a network."""
    to_return = "" if not add_newline else "\n"

    if not map_estimates:
        map_estimates = []
    to_return += "**Observations:**\n"

    for node in network.nodes.values():
        node_text = node.name if node.label is None else node.label
        node_value = "?" if node.name not in observations else observations[node.name]
        to_return += " - *{}* = {}\n".format(node_text, node_value)

    return to_return


def display_posteriors(variables, P_variables, add_newline=True):
    """Show probabilities for variables."""
    to_return = "" if not add_newline else "\n"
    to_return += "**Posteriors:**\n"

    for varname, prob_table in zip(variables, P_variables):
        prob = prob_table.iloc[1]["prob"]
        to_return += r"- $P(\text{{{}}} = T) = {:1.4f}$".format(varname, prob)
        to_return += "\n"

    return to_return


def display_MAP(network, observations, map_estimates, add_newline=True):
    """Show observed values of the nodes of a network."""
    to_return = "" if not add_newline else "\n"

    if not map_estimates:
        map_estimates = []
    to_return += "**State Estimate:**\n"

    for node in network.nodes.values():
        node_text = node.name if node.label is None else node.label

        if node.name in observations:
            node_value = "{} (Observation)".format(observations[node.name])
        elif node.name in map_estimates:
            node_value = "{} (Estimate)".format(observations[node.name])
        else:
            node_value = "?"

        to_return += " - *{}* = {}\n".format(node_text, node_value)

    return to_return


