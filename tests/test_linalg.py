from __future__ import division, print_function

import numpy as np
import random

from optprime import linalg

TOLERANCE = 1e-10


def test_o_matrix_is_orthonormal():
    dims = 10
    I = np.eye(dims)

    total_seeds = 100
    num_pure_rotations = 0

    for seed in range(total_seeds):
        rand = random.Random(seed)
        A = linalg.rand_o_matrix(dims, rand)

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

    v = linalg.orthogonalize(x, A)
    assert np.allclose(v, [0.0, 0.0, 0.0, 1.0], TOLERANCE)

def test_orthogonalize_with_non_orthonormal_input():
    # First 3 columns of a 4-dimensional identity matrix.
    A = np.eye(4)[:, :-1]
    A[0, 1] = 1
    x = np.array([1.0, 2.0, 3.0, 4.0])

    v = linalg.orthogonalize(x, A)
    assert np.allclose(v, [0.0, 0.0, 0.0, 1.0], TOLERANCE)

def test_orthogonalize_with_random_orthonormal_input():
    dims = 4
    total_seeds = 100

    x = np.array([1.0, 2.0, 3.0, 4.0])

    for seed in range(total_seeds):
        rand = random.Random(seed)
        q = linalg.rand_o_matrix(dims, rand)
        answer = q[:, -1]
        A = q[:, :-1]

        v = linalg.orthogonalize(x, A)
        assert (np.allclose(v, answer, TOLERANCE)
                or np.allclose(v, -answer, TOLERANCE))

def test_orthogonalize_with_random_non_orthonormal_input():
    dims = 4
    total_seeds = 100

    x = np.array([1.0, 2.0, 3.0, 4.0])

    for seed in range(total_seeds):
        rand = random.Random(seed)
        q = linalg.rand_o_matrix(dims, rand)
        answer = q[:, -1]
        A = q[:, :-1]
        A[:, 1] += 0.05 * A[:, 0]
        A[:, 2] += 0.04 * A[:, 0]
        A[:, 2] += 0.03 * A[:, 1]

        v = linalg.orthogonalize(x, A)
        assert (np.allclose(v, answer, TOLERANCE)
                or np.allclose(v, -answer, TOLERANCE))

