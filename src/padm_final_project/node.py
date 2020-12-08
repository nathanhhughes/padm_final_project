"""Module containing the Node class."""
import pandas as pd
import numpy as np
import random


class Node:
    """
    Represents a random variable in the Bayes network.

    Currently only supports binary random variables.

    Attributes:
      name (str): name of the node
      probabilities (pd.DataFrame): conditional probability table
      parents (List[Node]): all parents of the node
      label (Optional[str]): human readable name
    """

    name = None
    probabilities = None
    parents = None

    def __init__(self, name, probabilities, parents=None, label=None):
        """
        Construct a node directly from the conditional probability table (CPT).

        Args:
          name (str): node name
          probabilities (pd.DataFrame): table of parent values and conditional probabilities
          parents (Optional[List[Node]]): parents in the conditional probability table
          label (Optional[str]): human friendly name of the node
        """
        # for nodes without parents, probabilities is just [0.95], P(X=true)
        self.name = name
        self.label = label

        if parents is None:
            self.parents = []
        else:
            self.parents = parents

        self.probabilities = probabilities

    @classmethod
    def from_probabilities(cls, name, parents, probabilities, label=None):
        """
        Construct a node directly from the conditional probability table (CPT).

        Args:
          name (str): node name
          probabilities (Iterable[float]): likelihoods of node being true ordered by cartesian product of parents
          parents (Optional[List[Node]]): parents in the conditional probability table
          label (Optional[str]): human friendly name of the node
        """
        parent_strings = []

        for parent in parents:
            parent_strings += [parent.name]

        df = pd.DataFrame(columns=parent_strings + ["prob"])
        ar = [False] * len(parent_strings)
        # we need to create 2^parents rows - the cpt table.
        # these few lines generate all combinations of false / true statements
        for i in range(2 ** len(parent_strings)):
            df.loc[i] = ar + [probabilities[i]]
            j = 0
            while j < len(parent_strings) and True:
                if not ar[j]:
                    ar[j] = True
                    ar[0:j] = [False] * (j)
                    break
                else:
                    j += 1

        return cls(name, df, parents=parents, label=label)

    @classmethod
    def from_inhibitions(cls, name, parents, inhibition_values, label=None):
        """
        Construct a CPT from a list of inhbiting probabilties from parents.

        Requires parents to be independent of one another

        Args:
          name (str): node name
          parents (List[Node]): parents in the conditional probability table
          inhibition_values (List[float]): likelihood of node being false given
            parent is true, in order of parents provided

        """
        probabilities = []

        # Get parent assignment ordering
        bitmasks = []
        for i in range(2 ** len(parents)):
            bitmask = bin(i)[-1:1:-1]  # drop first two characters and reverse bit order
            if len(bitmask) < len(parents):
                bitmask += "0" * (len(parents) - len(bitmask))

            bitmasks.append(
                [True if bit == "1" else False for bit in bitmask[: len(parents)]]
            )

        # compute probabilities for all parent assignments assume independence between parents
        for bitmask in bitmasks:
            probability = 1.0
            for index, _ in enumerate(parents):
                if bitmask[index]:
                    probability *= inhibition_values[index]
            # this is the inhibition, so take the inverse
            probabilities.append(1.0 - probability)

        return cls.from_probabilities(name, parents, probabilities, label=label)

    def __str__(self):
        """Show the conditional probability table."""
        res = (
            "Node name: "
            + self.name
            + "\n"
            + "Node Parents: "
            + str([parent.name for parent in self.parents])
            + "\n"
            + str(self.probabilities)
        )
        res = res + "\n---------------------\n"
        return res

    def __repr__(self):
        """Show some basic info about the node."""
        return (
            "{name:"
            + self.name
            + ", probabilities:"
            + str(self.probabilities)
            + ", parents:"
            + str(self.parents)
            + "}"
        )

    def get_html_cpt(self):
        """
        Write the CPT as an html table.

        Note: primarily used for drawing a CPT diagram.

        Returns:
            str: html-like table of CPT that is valid for graphviz

        """
        html_body = '<<table border="0" cellborder="1">'
        html_body += '<tr><td colspan="{}"><b>Node: {}</b></td></tr>'.format(
            len(self.parents) + 1, self.name if self.label is None else self.label
        )

        if len(self.parents) == 0:
            html_body += "<tr><td>P({} = True) = {:1.3f}</td></tr>".format(
                self.name, self.probabilities.iloc[0]["prob"]
            )
            html_body += "</table>>"
            return html_body

        html_body += "<tr>"
        html_body += '<td colspan="{}">Parents</td>'.format(len(self.parents))
        html_body += '<td rowspan="2">P({} = True)</td>'.format(self.name)
        html_body += "</tr>"

        html_body += "<tr>"
        for column in self.probabilities.columns[:-1]:
            html_body += "<td>{}</td>".format(column)
        html_body += "</tr>"

        for row in self.probabilities.itertuples():
            html_body += "<tr>"
            for idx, column in enumerate(self.probabilities.columns):
                if idx == len(self.probabilities.columns) - 1:
                    html_body += "<td>{:1.3f}</td>".format(row[idx + 1])
                else:
                    html_body += "<td>{}</td>".format(row[idx + 1])
            html_body += "</tr>"

        html_body += "</table>>"
        return html_body
