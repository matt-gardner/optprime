from __future__ import division, print_function

import numpy as np
import random

from optprime.linalg import *

TOLERANCE = 1e-10


def test_o_matrix_is_orthonormal():
    dims = 10
    I = np.eye(dims)

    total_seeds = 100
    num_pure_rotations = 0

    for seed in range(total_seeds):
        rand = random.Random(seed)
        A = rand_o_matrix(dims, rand)

        # Note: numpy's dot() implements generalized matrix multiplication.
        product = np.dot(A, A.T)
        assert np.allclose(product, I, TOLERANCE), (
                'Transpose is not inverse (seed {})'.format(seed))

        eigenvalues = np.linalg.eigvals(A)
        assert np.allclose(abs(eigenvalues), 1.0, TOLERANCE), (
                'Non-unit eigenvalues (seed {})'.format(seed))

        # If it's a pure rotation, there will be no real eigenvalues.  If it's
        # a rotation + a reflection, there will be two real eigenvalues.
        real_eigs = sum(1 for x in eigenvalues.imag if abs(x) < TOLERANCE)
        assert real_eigs in (0, 2), (
                'Number of real eigenvalues not 0 or 2 (seed {})'.format(seed))

        if not real_eigs:
            num_pure_rotations += 1

    assert abs(num_pure_rotations / total_seeds - 0.5) < 0.05

def test_orthogonalize_with_orthonormal_input():
    # First 3 columns of a 4-dimensional identity matrix.
    A = np.eye(4)[:, :-1]
    x = np.array([1.0, 2.0, 3.0, 4.0])

    v = orthogonalize(x, A)
    assert np.allclose(v, [0.0, 0.0, 0.0, 1.0], TOLERANCE)

def test_orthogonalize_with_non_orthonormal_input():
    # First 3 columns of a 4-dimensional identity matrix.
    A = np.eye(4)[:, :-1]
    A[0, 1] = 1
    x = np.array([1.0, 2.0, 3.0, 4.0])

    v = orthogonalize(x, A)
    assert np.allclose(v, [0.0, 0.0, 0.0, 1.0], TOLERANCE)

def test_orthogonalize_with_random_orthonormal_input():
    dims = 4
    total_seeds = 100

    x = np.array([1.0, 2.0, 3.0, 4.0])

    for seed in range(total_seeds):
        rand = random.Random(seed)
        q = rand_o_matrix(dims, rand)
        answer = q[:, -1]
        A = q[:, :-1]

        v = orthogonalize(x, A)
        assert (np.allclose(v, answer, TOLERANCE)
                or np.allclose(v, -answer, TOLERANCE))

def test_orthogonalize_with_random_non_orthonormal_input():
    dims = 4
    total_seeds = 100

    x = np.array([1.0, 2.0, 3.0, 4.0])

    for seed in range(total_seeds):
        rand = random.Random(seed)
        q = rand_o_matrix(dims, rand)
        answer = q[:, -1]
        A = q[:, :-1]
        A[:, 1] += 0.05 * A[:, 0]
        A[:, 2] += 0.04 * A[:, 0]
        A[:, 2] += 0.03 * A[:, 1]

        v = orthogonalize(x, A)
        assert (np.allclose(v, answer, TOLERANCE)
                or np.allclose(v, -answer, TOLERANCE))

def test_eigh_sorted():
    # Note that this test array is symmetric.
    A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    eigvals, eigvecs = eigh_sorted(A)

    actual_eigvals = np.array([11.34481428, 0.17091519, -0.51572947])
    assert np.allclose(eigvals, actual_eigvals, TOLERANCE)
    actual_eigvec1 = np.array([-0.32798528, -0.59100905, -0.73697623])
    assert np.allclose(eigvecs[:, 0], actual_eigvec1, TOLERANCE)
    actual_eigvec2 = np.array([0.59100905, -0.73697623, 0.32798528])
    assert np.allclose(eigvecs[:, 1], actual_eigvec2, TOLERANCE)
    actual_eigvec3 = np.array([-0.73697623, -0.32798528, 0.59100905])
    assert np.allclose(eigvecs[:, 2], actual_eigvec3, TOLERANCE)

def test_eigh_sorted_neg():
    # Note that this test array is symmetric.
    A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    eigvals, eigvecs = eigh_sorted(A)

    actual_eigvals = np.array([1, 1, -1])
    assert np.allclose(eigvals, actual_eigvals, TOLERANCE)

def test_bingham_sampler_init():
    # Note that this test array is symmetric.
    A = np.array([[5, 0, 0], [0, 3, 0], [0, 0, 1]])

    bs = BinghamSampler(A)
    assert list(bs._lambdas) == [4, 2]

def test_bingham_sampler_dual():
    # Note that this test array is symmetric.
    A = np.array([[5, 0, 0], [0, 3, 0], [0, 0, 1]])

    bs = BinghamSampler(A)
    bs_dual1 = BinghamSampler(-A)
    bs_dual2 = bs.dual()

    assert np.allclose(bs_dual1._lambdas, bs_dual2._lambdas)
    assert np.allclose(bs_dual1._eigvecs, bs_dual2._eigvecs)

def test_bingham_sampler_init_neg():
    # Note that this test array is symmetric.
    A = np.array([[-3, 0, 0], [0, 5, 0], [0, 0, -1]])

    bs = BinghamSampler(A)
    assert list(bs._lambdas) == [8, 6]

def test_complex_bingham_sampler_init():
    # Note that this test array is symmetric.
    A = np.array([[5, 0, 0], [0, 3, 0], [0, 0, 1]])

    bs = ComplexBinghamSampler(A)
    assert list(bs._lambdas) == [4, 2]

def test_complex_bingham_sampler_dual():
    # Note that this test array is symmetric.
    A = np.array([[5, 0, 0], [0, 3, 0], [0, 0, 1]])

    bs = ComplexBinghamSampler(A)
    bs_dual1 = ComplexBinghamSampler(-A)
    bs_dual2 = bs.dual()

    assert np.allclose(bs_dual1._lambdas, bs_dual2._lambdas)
    assert np.allclose(bs_dual1._eigvecs, bs_dual2._eigvecs)

def test_complex_bingham_sampler_init_neg():
    # Note that this test array is symmetric.
    A = np.array([[-3, 0, 0], [0, 5, 0], [0, 0, -1]])

    bs = ComplexBinghamSampler(A)
    assert list(bs._lambdas) == [8, 6]

def test_complex_bingham_pick_sampler1():
    """Values from Table 2 (Kent, Constable, and Er, 2004)."""
    lambdas = [0.01, 0.01]
    bs = ComplexBinghamSampler(lambdas=lambdas)
    assert bs._pick_sampler() == bs.sample_m2

def test_complex_bingham_pick_sampler2():
    """Values from Table 2 (Kent, Constable, and Er, 2004)."""
    lambdas = [0.1, 0.01]
    bs = ComplexBinghamSampler(lambdas=lambdas)
    assert bs._pick_sampler() == bs.sample_m2

def test_complex_bingham_pick_sampler3():
    """Values from Table 2 (Kent, Constable, and Er, 2004)."""
    lambdas = [0.5, 0.01]
    bs = ComplexBinghamSampler(lambdas=lambdas)
    assert bs._pick_sampler() == bs.sample_m2

def test_complex_bingham_pick_sampler4():
    """Values from Table 2 (Kent, Constable, and Er, 2004)."""
    lambdas = [1.0, 0.01]
    bs = ComplexBinghamSampler(lambdas=lambdas)
    assert bs._pick_sampler() == bs.sample_m1

def test_complex_bingham_pick_sampler5():
    """Values from Table 2 (Kent, Constable, and Er, 2004)."""
    lambdas = [0.1, 0.1]
    bs = ComplexBinghamSampler(lambdas=lambdas)
    assert bs._pick_sampler() == bs.sample_m2

def test_complex_bingham_pick_sampler6():
    """Values from Table 2 (Kent, Constable, and Er, 2004)."""
    lambdas = [1.0, 0.1]
    bs = ComplexBinghamSampler(lambdas=lambdas)
    assert bs._pick_sampler() == bs.sample_m1

def test_complex_bingham_pick_sampler7():
    """Values from Table 2 (Kent, Constable, and Er, 2004)."""
    lambdas = [0.5, 0.5]
    bs = ComplexBinghamSampler(lambdas=lambdas)
    assert bs._pick_sampler() == bs.sample_m1

def test_von_mises_fisher_norm():
    """Ensure that samples are on the unit sphere."""
    for i in range(10):
        z = sample_von_mises_fisher(5, i, random)
        norm = sum(z**2)
        assert abs(norm - 1) < TOLERANCE

def test_von_mises_fisher_uniform():
    """Ensure that the mean is the origin."""
    samples = [sample_von_mises_fisher(5, 0, random) for _ in range(10000)]

    mean = sum(samples) / len(samples)
    assert np.allclose(mean, 0, atol=0.02)

def test_von_mises_fisher_nonuniform():
    """Ensure that samples are biased towards the mean."""
    for i in range(10, 100):
        for _ in range(10):
            z = sample_von_mises_fisher(5, i, random)
            assert np.all(abs(z) <= z[0])

def test_chol_update():
    rand = random.Random(42)

    # Create an initial positive definite matrix A.
    A = np.zeros((4, 4))
    for i in range(1, 5):
        x = np.array([rand.normalvariate(0, 1) for _ in range(4)])
        x /= np.linalg.norm(x)
        A += np.outer(x, x)

    # Find the lower-triangular Cholesky decomposition of A.
    L_orig = np.linalg.cholesky(A)

    # Create a new data point.
    y = np.array([rand.normalvariate(0, 1) for _ in range(4)])
    y /= np.linalg.norm(y)
    scatter = np.outer(y, y)

    L_boring = np.linalg.cholesky(A + scatter)

    L_fancy = L_orig.copy()
    chol_update(L_fancy, y)
    assert np.allclose(L_fancy, L_boring)

    chol_downdate(L_fancy, y)
    assert np.allclose(L_fancy, L_orig)

def test_bingham_uniform_constant():
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    bs = BinghamSampler(A)
    # Note: c represents the area of the unit 2-sphere (4 pi).
    c = math.exp(bs.log_const())
    assert abs(c - 4 * math.pi) < 0.01
