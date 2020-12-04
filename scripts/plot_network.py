#!/usr/bin/env python
"""Generate a draw a network."""
from padm_final_project.network_utils import generate_network
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
import pygraphviz as pgv
import networkx as nx
import numpy as np
import tempfile
import sys



def main():
    """Generate a network and draw it."""
    a = generate_network()

    with tempfile.NamedTemporaryFile() as fout:
        fout.write(a.draw_cpt_diagram())
        fout.seek(0)
        img_to_plot = mpimage.imread(fout.name)
        plt.imshow(img_to_plot)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
