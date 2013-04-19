"""Miscellaneous linear algebra operations.

For numpy arrays vs. matrices, see:
    http://www.scipy.org/NumPy_for_Matlab_Users#head-e9a492daa18afcd86e84e07cd2824a9b1b651935
"""

from __future__ import division, print_function

import itertools
import math
import random

try:
    import numpy as np
except ImportError:
    import numpypy as np

def rand_o_matrix(n, rand=None):
    """Creates a random Haar-distributed orthonormal matrix.

    Uses the Stewart algorithm.
    """
    if rand is None:
        rand = random

    # `A` holds the orthonormal matrix that is built inductively (starting
    # from [[1.0]] or [[-1.0]], which are the only 1x1 orthonormal matrices.
    A = np.zeros((n, n))
    A[0, 0] = rand.randrange(-1, 2, 2)

    # Identity matrices can come in handy.
    I = np.eye(n)

    # `x` holds an n-dimensional vector (an nx1 matrix) that will hold normal
    # vectors that will define planes in successively larger subspaces.
    x = np.zeros((n, 1))

    for i in range(1, n):
        # The m-dimensional subspace (in the top-left corner) is the relevant
        # space during the current iteration.
        m = i + 1

        # Extend A to m dimensions by adding a new orthogonal basis vector.
        # The value is either [0, 0, ..., 1] or [0, 0, ..., -1] (flipped).
        # If you don't ever flip, then the algorithm only produces
        # reflection+rotations (never pure rotations).
        A[i, i] = rand.randrange(-1, 2, 2)

        # Numpy "broadcasting" converts a single-axis array into an mx1 array,
        # which then gets saved to the first m slots of x.
        x[:m] = rand_norm_array(i + 1, rand)[:, None]

        # Reflect A across the plane defined by x (this is a Householder
        # transformation).
        householder_mat = I - 2 * np.dot(x, x.T)
        A = np.dot(householder_mat, A)

    return A

def rand_norm_array(n, rand=None):
    """Creates a random normalized array of given size."""
    if rand is None:
        rand = random

    samples = np.empty(n, 'float')
    for i in range(n):
        samples[i] = rand.normalvariate(0, 1)
    normalization = (samples ** 2).sum() ** 0.5
    return samples / normalization

def rand_cliques_matrix(n, m, rand=None):
    """Create an nxn orthogonal matrix with m random orthogonal submatrices.

    Note that the matrix is returned as a numpy array (not a numpy matrix).
    """
    if rand is None:
        rand = random

    result = np.zeros((n, n))
    min_block_size = n // m
    num_big_blocks = n % m

    index = 0
    for block_id in range(m):
        if block_id < num_big_blocks:
            N = min_block_size + 1
        else:
            N = min_block_size

        submatrix = rand_o_matrix(N, rand)
        result[index:index + N, index:index + N] = submatrix

        index += N

    return result

def rand_perm_matrix(n, rand=None):
    """Create a random permutation matrix.

    Note that the matrix is returned as a numpy array (not a numpy matrix).
    """
    if rand is None:
        rand = random

    variables = list(range(n))
    rand.shuffle(variables)
    A = np.zeros((n, n))
    for i, j in enumerate(variables):
        A[i, j] = 1
    return A

def orthogonalize(x, A):
    """Orthogonalize the vector x relative to A, a matrix of column vectors.

    The input is expected to be a numpy array (not a numpy matrix).  The
    array A need not be an orthonormal matrix.

    Based on the Modified Gram-Schmidt algorithm (numerically stable).
    """
    v = x.copy()
    q, _ = np.linalg.qr(A)
    for u in q.T:
        # Subtract out the projection of v onto u.
        v -= np.dot(v, u) * u
    # Normalize.
    v /= np.sqrt(np.dot(v, v))
    return v

def kume_walker_phi(p, M, N):
    """Return the truncation error estimate from Kume-Walker.

    M = max{a_i; i=1,...,p, |b_i|; i=0,...,p}

    Phi is a bound on the sum of all of the w(l) where \sum l_i >= N.
    """
    log_tau = p * math.lgamma(0.5) - math.lgamma(p / 2 + 0.5)
    log_phi = (log_tau
            + N * (math.log(M) + math.log(1 + (p + 1) ** 0.5))
            - math.lgamma(N + 1)
            + M * (1 + (p + 1) ** 0.5))
    return math.exp(log_phi)

def kume_walker_phi2(p, M, N):
    """Return the truncation error estimate from Kume-Walker (modified).

    M = max{a_i; i=1,...,p, |b_i|; i=0,...,p}

    Phi is a bound on the sum of all of the w(l) where \sum l_i >= N.
    """
    log_tau = p * math.lgamma(0.5) - math.lgamma(p / 2 + 0.5)
    log_phi = (log_tau
            + N * (math.log(M) + 0.5 * math.log(p + 1))
            - math.lgamma(N + 1)
            + M * (p + 1) ** 0.5)
    return math.exp(log_phi)

def kume_walker_w(l, a):
    log_w = math.lgamma(0.5)
    log_w -= math.lgamma(0.5 + sum((l_i + 0.5) for l_i in l))
    for l_i, a_i in zip(l, a):
        log_w += l_i * math.log(a_i) + math.lgamma(l_i + 0.5)
        log_w -= math.lgamma(l_i + 1)
    return math.exp(log_w)

def kume_walker_norm(a, N):
    """Sum up all of the w(l) terms with \sum l_i < N."""
    indices = range(len(a))
    total = 0
    for cap in range(N):
        print('CAP:', cap)
        #print(list(itertools.combinations_with_replacement(indices, cap)))
        for combos in itertools.combinations_with_replacement(indices, cap):
            l = [0] * len(a)
            for i in combos:
                l[i] += 1
            total += kume_walker_w(l, a)
            print(l)
        print('total:', total)
    return total

def kume_walker_norm2(a_i, p, N):
    """Sum up all of the w(l) terms with \sum l_i < N.

    This variant assumes that a_i = a_1 for all i.  Note that p is the number
    of dimensions.
    """
    w_i_terms = []
    for l_i in range(N):
        # The log of the term of w(l) corresponding to a single i (within the
        # product).
        term = (l_i * math.log(a_i) + math.lgamma(l_i + 0.5)
                - math.lgamma(l_i + 1))
        w_i_terms.append(term)

    total = 0
    for cap in range(N):
        print('CAP:', cap)
        # The log of the base term (outside the product) in w(l)
        base_term = math.lgamma(0.5) - math.lgamma(0.5 * (p + 1) + cap)

        for part in partitions(cap):
            if len(part) > p:
                continue
            print(part)

            log_w = base_term
            # Note that partitions(0) returns [0] (for better or worse).
            if part[0] == 0:
                part_size = 0
            else:
                for i, l_i in enumerate(part):
                    log_w += w_i_terms[l_i]
                part_size = i + 1
            # Add in all of the 0 terms.
            log_w += (p - part_size) * w_i_terms[0]

            # The binomial coefficient (choose function) indicates the
            # number of identical terms (i.e., the number of l vectors
            # that can be formed by rearranging this partition).
            log_count = (math.lgamma(p + 1) - math.lgamma(part_size + 1)
                    - math.lgamma(p - part_size + 1))
            total += math.exp(log_w + log_count)
        print('total:', total)

    return total

def partitions(n):
    """Generate all integer partitions of n (accelAsc)

    Kelleher and O'Sullivan. Generating All Partitions: A Comparison of Two
    Encodings. 2009.
    See: http://homepages.ed.ac.uk/jkellehe/partitions.php

    See http://arxiv.org/abs/0909.2331
    """
    a = [0] * (n + 1)
    k = 1
    a[0] = 0
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2*x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]
# vim: et sw=4 sts=4
