import numpy as np

from neweds.core.generator import generate_coupled_system


def test_generate_coupled_system_is_reproducible():
    a = generate_coupled_system(seed=42)
    b = generate_coupled_system(seed=42)
    assert a.equals(b)


def test_generate_coupled_system_does_not_mutate_global_numpy_rng():
    np.random.seed(123)
    before = np.random.random(3)

    np.random.seed(123)
    _ = generate_coupled_system(seed=42)
    after = np.random.random(3)

    assert np.allclose(before, after)
