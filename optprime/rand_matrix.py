#!/usr/bin/env python

import random

try:
    from numpy import empty, eye, matrix, zeros
except ImportError:
    from numpypy import empty, eye, matrix, zeros

def rand_o_matrix(n, rand=None):
    """Creates a random Haar-distributed orthonormal matrix.

    Uses the Stewart algorithm.
    """
    # `A` holds the orthonormal matrix that is built inductively (starting
    # from [[1.0]], which is a 1x1 orthonormal matrix.
    A = matrix(zeros((n, n)))
    A[0, 0] = 1.0

    # Identity matrices can come in handy.
    I = matrix(eye(n))

    # `x` holds an n-dimensional vector (an nx1 matrix) that will hold normal
    # vectors that will define planes in successively larger subspaces.
    x = matrix(zeros((n, 1)))

    for i in range(1, n):
        # The m-dimensional subspace (in the top-left corner) is the relevant
        # space during the current iteration.
        m = i + 1

        # Extend A to m dimensions by adding a new orthogonal basis vector.
        A[i, i] = 1.0

        # Numpy "broadcasting" converts a single-axis array into an mx1 array,
        # which then gets saved to the first m slots of x.
        x[:m] = rand_norm_array(i + 1)[:, None]

        # Reflect A across the plane defined by x (this is a Householder
        # transformation).
        A = (I - 2 * x * x.T) * A

    return A

def rand_norm_array(n, rand=None):
    """Creates a random normalized array of given size."""
    if rand is None:
        rand = random

    samples = empty(n, 'float')
    for i in range(n):
        samples[i] = rand.normalvariate(0, 1)
    normalization = (samples ** 2).sum() ** 0.5
    return samples / normalization

# vim: et sw=4 sts=4
