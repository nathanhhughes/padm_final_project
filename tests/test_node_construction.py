from padm_final_project.node import Node
import pytest


def test_from_probabilities():
    a = Node("a", 1.0)
    b = Node("b", 1.0)
    test_node = Node.from_probabilities("test", [a, b], [1.0, 0.1, 0.2, 0.02])

    # first value: a and b are false
    assert not test_node.probabilities.iloc[0]["a"]
    assert not test_node.probabilities.iloc[0]["b"]
    assert pytest.approx(test_node.probabilities.iloc[0]["prob"], 1.0)
    # second value: a is true, b is false
    assert test_node.probabilities.iloc[1]["a"]
    assert not test_node.probabilities.iloc[1]["b"]
    assert pytest.approx(test_node.probabilities.iloc[0]["prob"], 0.1)
    # third value: a is false, b is true
    assert not test_node.probabilities.iloc[2]["a"]
    assert test_node.probabilities.iloc[2]["b"]
    assert pytest.approx(test_node.probabilities.iloc[0]["prob"], 0.2)
    # fourth value: a and b are true
    assert test_node.probabilities.iloc[3]["a"]
    assert test_node.probabilities.iloc[3]["b"]
    assert pytest.approx(test_node.probabilities.iloc[0]["prob"], 0.02)


def test_from_inhibitions():
    """Test that we can generate a good CPT."""
    a = Node("a", 1.0)
    b = Node("b", 1.0)
    test_node = Node.from_inhibitions("test", [a, b], [0.1, 0.2])

    assert pytest.approx(test_node.probabilities.iloc[0]["prob"], 0.0)
    assert pytest.approx(test_node.probabilities.iloc[1]["prob"], 0.9)
    assert pytest.approx(test_node.probabilities.iloc[2]["prob"], 0.8)
    assert pytest.approx(test_node.probabilities.iloc[3]["prob"], 0.98)
