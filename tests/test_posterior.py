from padm_final_project.network_utils import make_sample_network
import random
import pytest



@pytest.mark.filterwarnings("ignore")
def test_posterior_tree_network():
    """Show the map."""
    net = make_sample_network(10, num_extra_branches=0)
    observations = {'n0': False, 'n1': False, 'n3': True, 'n5': False, 'n6': False, 'n8': False, 'n9': False}
    queries = ['n4', 'n2', 'n7']
    for query in queries:
        prob = net.get_posterior([query], observations)
        assert prob <= 1.0
        assert prob >= 0.0
