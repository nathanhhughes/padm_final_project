"""Tests around making a network."""
from padm_final_project.network_utils import generate_network


def test_generate_network_basic():
    """Make a network and make sure nothing goes wrong."""
    net = generate_network()
    assert net is not None


def test_draw_network():
    """Draw a network and make sure nothing goes wrong."""
    net = generate_network()
    net.draw_net()
    assert net is not None
