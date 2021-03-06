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

def test_eigh_swapped():
    # Note that this test array is symmetric.
    A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    eigvals, eigvecs = eigh_swapped(A)

    assert np.allclose(eigvals[-1], -0.51572947, TOLERANCE)
    actual_eigvec3 = np.array([-0.73697623, -0.32798528, 0.59100905])
    assert np.allclose(eigvecs[:, -1], actual_eigvec3, TOLERANCE)

def test_eigh_sorted_neg():
    # Note that this test array is symmetric.
    A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    eigvals, eigvecs = eigh_sorted(A)

    actual_eigvals = np.array([1, 1, -1])
    assert np.allclose(eigvals, actual_eigvals, TOLERANCE)

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
