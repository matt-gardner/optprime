from __future__ import division, print_function

import numpy as np
import random

from optprime.dirstats import *

TOLERANCE = 1e-10


def test_bingham_sampler_init():
    # Note that this test array is symmetric.
    A = np.array([[5, 0, 0], [0, 3, 0], [0, 0, 1]])

    bs = bingham_sampler_from_matrix(A)
    assert list(bs._lambdas) == [4, 2]

def test_bingham_sampler_dual():
    # Note that this test array is symmetric.
    A = np.array([[5, 0, 0], [0, 3, 0], [0, 0, 1]])

    bs = bingham_sampler_from_matrix(A)
    bs_dual1 = bingham_sampler_from_matrix(-A)
    bs_dual2 = bs.dual()

    assert np.allclose(bs_dual1._lambdas, bs_dual2._lambdas)
    assert np.allclose(bs_dual1._eigvecs, bs_dual2._eigvecs)

def test_bingham_sampler_init_neg():
    # Note that this test array is symmetric.
    A = np.array([[-3, 0, 0], [0, 5, 0], [0, 0, -1]])

    bs = bingham_sampler_from_matrix(A)
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

def test_inverse_log_bingham_const():
    eigvals = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    log_const = log_bingham_const_eigvals(eigvals)

    target = log_const - 5
    index = 3
    inv = inverse_log_bingham_const(eigvals, target, index)

    eigvals[index] = inv
    assert np.allclose(log_bingham_const_eigvals(eigvals), target)

def test_log_bingham_const_swap():
    eigvals = np.array([10.0, 0.0, 0.0])
    log_const_1 = log_bingham_const_eigvals(eigvals)

    eigvals = np.array([0.0, 0.0, 10.0])
    log_const_2 = log_bingham_const_eigvals(eigvals)

    assert np.allclose(log_const_1, log_const_2)

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
    for kappa in (20, 25, 30, 40):
        for _ in range(10):
            z = sample_von_mises_fisher(5, kappa, random)
            assert np.all(abs(z) <= z[0])

def test_bingham_uniform_eigvals_constant():
    log_const = log_bingham_const_eigvals(np.array([0.0, 0.0, 0.0]))
    # Note: c represents the area of the unit 2-sphere (4 pi).
    c = math.exp(log_const)
    assert abs(c - 4 * math.pi) < 0.01

def test_bingham_uniform_matrix_constant():
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    bs = bingham_sampler_from_matrix(A)
    # Note: c represents the area of the unit 2-sphere (4 pi).
    # The -1 term in exp is because of the exp(x^T A x) term where x^T A x = 1.
    c = math.exp(bs.log_const() - 1)
    assert abs(c - 4 * math.pi) < 0.01

def test_unionate():
    intervals = [(0, 4), (3, 5), (6, 8), (7.9, 10), (11, 12)]
    assert unionate(intervals) == [(0, 5), (6, 10), (11, 12)]

def test_unionate_unsorted():
    intervals = [(7.9, 10), (3, 5), (6, 8), (11, 12), (0, 4)]
    assert unionate(intervals) == [(0, 5), (6, 10), (11, 12)]

def test_unionate_unstaggered():
    intervals = [(0, 4), (2, 3), (6, 10), (7, 9), (8, 11)]
    assert unionate(intervals) == [(0, 4), (6, 11)]

def test_complement():
    inf = float('inf')
    intervals = [(0, 4), (3, 5), (6, 8), (7.9, 10), (11, 12)]
    assert complement(intervals) == [(-inf, 0), (5, 6), (10, 11), (12, inf)]

def test_complement_neg_inf():
    inf = float('inf')
    intervals = [(-inf, 5), (7, 30)]
    assert complement(intervals) == [(5, 7), (30, inf)]

def test_complement_pos_inf():
    inf = float('inf')
    intervals = [(13, 14), (15, 20), (23, inf)]
    assert complement(intervals) == [(-inf, 13), (14, 15), (20, 23)]
