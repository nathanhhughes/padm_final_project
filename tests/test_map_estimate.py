from padm_final_project.node import Node
from padm_final_project.bayes_net import BayesNet
import pytest


def make_network():
    """Make the example network from the notebook."""
    attack_1 = Node.from_probabilities("a1", [0.40])
    attack_2 = Node.from_probabilities("a2", [0.20])
    subsystem_1 = Node.from_probabilities("s1", [0.2, 0.7], parents=[attack_1])
    workstation_1 = Node.from_inhibitions("w1", [attack_2, subsystem_1], [0.6, 0.7])
    workstation_2 = Node.from_probabilities("w2", [0.1, 0.8], parents=[attack_2])
    return BayesNet([attack_1, attack_2, subsystem_1, workstation_1, workstation_2])


@pytest.mark.filterwarnings("ignore")
def test_map_first_day():
    """Show the map."""
    net = make_network()
    observations = {"w1": False, "w2": True}
    query = ["a1", "a2", "s1"]
    map_result = net.get_MAP_estimate(query, observations)
    assert map_result["a1"] == False
    assert map_result["a2"] == True
    assert map_result["s1"] == False


@pytest.mark.filterwarnings("ignore")
def test_map_second_day():
    """Show the map."""
    net = make_network()
    observations = {"w2": False, "s1": True}
    query = ["a1", "a2", "w1"]
    map_result = net.get_MAP_estimate(query, observations)
    assert map_result["a1"] == True
    assert map_result["a2"] == False
    assert map_result["w1"] == False
