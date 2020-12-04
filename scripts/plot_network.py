#!/usr/bin/env python
"""Generate a draw a network."""
from padm_final_project.network_utils import generate_network
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
import tempfile


def main():
    """Generate a network and draw it."""
    a = generate_network()

    fig, ax = plt.subplots(2)
    a.draw_net(ax=ax[0])

    with tempfile.NamedTemporaryFile() as fout:
        fout.write(a.draw_cpt_diagram().draw(format="png", prog="dot"))
        fout.seek(0)
        img_to_plot = mpimage.imread(fout.name)

    ax[1].imshow(img_to_plot)
    ax[1].axis("off")
    plt.show()


if __name__ == "__main__":
    main()
