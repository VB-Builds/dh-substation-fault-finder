import pytest

from dhfault.sim import SimConfig, simulate_network


@pytest.fixture(scope="session")
def small_net():
    """24 simulated substations, one of each of the 8 fault types, 2022 history + 2023 test year."""
    return simulate_network(SimConfig(n_substations=24, faults_per_type=1, seed=5))
