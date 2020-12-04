#!/usr/bin/env python
"""Generate a draw a network."""
from padm_final_project.network_utils import generate_network
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
import tempfile
import pathlib


def main():
    """Generate a network and draw it."""
    a = generate_network()

    fig, ax = plt.subplots(2)
    a.draw_net(ax=ax[0])

    with tempfile.TemporaryDirectory() as output_dir:
        output_path = pathlib.Path(output_dir)
        gv_graph = a.get_cpt_diagram()
        gv_graph.render(directory=str(output_path))

        img_to_plot = mpimage.imread(str(output_path / "Digraph.gv.png"))

    ax[1].imshow(img_to_plot)
    ax[1].axis("off")
    plt.show()


if __name__ == "__main__":
    main()
