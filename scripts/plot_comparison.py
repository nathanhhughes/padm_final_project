#!/usr/bin/env python3
"""Quick script to make testing comparisons faster."""
from padm_final_project.network_utils import generate_random_cpt, make_sample_network
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import contextlib
import warnings
import random
import tqdm
import time


@contextlib.contextmanager
def time_routine(stored_times, name):
    """Context manager to measure elapsed time."""
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    if name not in stored_times:
        stored_times[name] = []
    stored_times[name].append(elapsed)


def get_random_assignment(true_prob=0.24):
    """Make a random assignment."""
    return np.random.uniform() <= true_prob


def get_input(network, observation_percent=0.70):
    """Get some random queries and observations."""

    def _no_parents(node):
        for potential_child in network.nodes:
            parents = [parent.name for parent in network.nodes[potential_child].parents]
            if node in parents:
                return False

        return True

    potential_obs = [
        node for node in list(network.nodes.keys())[1:] if _no_parents(node)
    ]
    num_obs = int(observation_percent * len(network.nodes))
    num_obs = len(potential_obs) if len(potential_obs) < num_obs else num_obs
    observations = {
        node: get_random_assignment() for node in random.sample(potential_obs, num_obs)
    }

    return list(network.nodes.keys())[0], observations


def main():
    """Make plots."""
    warnings.simplefilter("ignore")
    sizes = np.arange(5, 25, 5)
    num_extra_branches = np.arange(0, 4, 1)
    trials = 5

    size_timing_results = {}
    for size in tqdm.tqdm(sizes, desc="Running size trials"):
        for _ in range(trials):
            net = make_sample_network(size)
            query, observations = get_input(net)

            with time_routine(size_timing_results, size):
                net.get_posterior([query], observations)

    sparsity_timing_results = {}
    for branches in tqdm.tqdm(num_extra_branches, desc="Running sparsity trials"):
        for _ in range(trials):
            net = make_sample_network(11, num_extra_branches=branches)
            query, observations = get_input(net)

            with time_routine(sparsity_timing_results, branches):
                net.get_posterior([query], observations)

    sns.set()
    fig, ax = plt.subplots(1, 2)
    size_x = [int(key) for key in size_timing_results]
    size_y = [np.mean(size_timing_results[key]) for key in size_timing_results]
    size_y_std = [np.std(size_timing_results[key]) for key in size_timing_results]
    ax[0].errorbar(size_x, size_y, size_y_std, marker="o")
    ax[0].set_xlabel("Number of nodes")
    ax[0].set_ylabel("Time elapsed [s]")
    ax[0].set_title("Average Posterior Computation Time vs. Number of Nodes")

    sparsity_x = [int(key) for key in sparsity_timing_results]
    sparsity_y = [
        np.mean(sparsity_timing_results[key]) for key in sparsity_timing_results
    ]
    sparsity_y_std = [
        np.std(sparsity_timing_results[key]) for key in sparsity_timing_results
    ]
    ax[1].errorbar(sparsity_x, sparsity_y, sparsity_y_std, marker="o")
    ax[1].set_xlabel("Sparseness")
    ax[1].set_ylabel("Time elapsed [s]")
    ax[1].set_title("Average Posterior Computation Time vs. Network Sparseness")

    plt.tight_layout()
    fig.set_size_inches([15, 6])
    plt.show()


if __name__ == "__main__":
    main()
