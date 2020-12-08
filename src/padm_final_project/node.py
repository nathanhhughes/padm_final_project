"""Module containing the Node class."""
# Current assumption: nodes can be either true or false (rather than A B C)
# Current assumption: store only the probability that the nodes is true
import pandas as pd
import numpy as np
import random

class Node:
    """
    Represents a random variable in the Bayes network.

    Attributes:
      name (str): name of the node
      probabilities (pd.DataFrame): conditional probability table
      parents (List[Node]): all parents of the node
    """

    name = None
    probabilities = None
    parents = None

    # TODO(nathan) think about whether optional probabilities make sense

    def __init__(self, name, probabilities, parents=None):
        """
        Construct a node directly from the conditional probability table (CPT).

        Args:
          name (str): node name
          probabilities (pd.DataFrame): likelihoods of node being true ordered by cartesian product of parents
          probabilities (np.array): likelihoods of node being true ordered by cartesian product of parents
          parents (Optional[List[Node]]): parents in the conditional probability table
        """
        # for nodes without parents, probabilities is just [0.95], P(X=true)
        self.name = name
        self.parents = parents
        if parents is None:
            self.parents = []
        
        if isinstance(probabilities, pd.DataFrame):
            self.probabilities = probabilities
        if isinstance(probabilities, (np.ndarray, np.generic)):
            parent_strings = []
            for parent in self.parents:
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
            self.probabilities = df
            

    @classmethod
    def from_inhibitions(cls, name, parents, inhibition_values):
        """
        Construct a CPT from a list of inhbiting probabilties from parents.

        Args:
          name (str): node name
          parents (List[Node]): parents in the conditional probability table
          inhibition_values (List[float]): likelihood of node being false given
            parent is true, in order of parents provided

        """
        raise NotImplementedError()

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
        if len(self.parents) == 0:
            return "P({} = True) = {:1.3f}".format(self.name, self.probabilities.iloc[0]["prob"])
        else:
            html_body = '<table border="0" cellborder="1"><tr>'
            for column in self.probabilities.columns[:-1]:
                html_body += "<td>{}</td>".format(column)
            html_body += "<td>P({} = True)</td>".format(self.name)
            html_body += "</tr>"

            for row in self.probabilities.itertuples():
                html_body += "<tr>"
                for idx, column in enumerate(self.probabilities.columns):
                    if idx == len(self.probabilities.columns) - 1:
                        html_body += "<td>{:1.3f}</td>".format(row[idx + 1])
                    else:
                        html_body += "<td>{}</td>".format(row[idx + 1])
                html_body += "</tr>"

            html_body += "</table>"
            return "<{}>".format(html_body)
