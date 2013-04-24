"""Miscellaneous linear algebra operations.

For numpy arrays vs. matrices, see:
    http://www.scipy.org/NumPy_for_Matlab_Users#head-e9a492daa18afcd86e84e07cd2824a9b1b651935
"""

from __future__ import division, print_function

import bisect
import itertools
import math
import numpy
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
    for sum_l in range(N):
        for combos in itertools.combinations_with_replacement(indices, sum_l):
            l = [0] * len(a)
            for i in combos:
                l[i] += 1
            total += kume_walker_w(l, a)
    return total


class BinghamSampler(object):
    """Create a mixture of Dirichlets for all w(l) terms with \sum l_i < N."""
    def __init__(self, a_i, p, N):
        self.total = 0
        self.weights = []
        self.dirichlets = []

        for sum_l in range(N):
            for part in partitions(sum_l):
                if len(part) > p:
                    continue

                d = KumeWalkerDirichlet(part, a_i, p)
                weight = d.weight()
                self.total += weight

                self.weights.append(self.total)
                self.dirichlets.append(d)

        # TODO: Consider sorting and pruning low-weight elements.

    def sample(self, rand):
        u = rand.uniform(0, self.total)
        i = bisect.bisect_left(self.weights, u)
        dirichlet = self.dirichlets[i]
        return dirichlet.sample(rand)


class KumeWalkerDirichlet(object):
    """A sampler for a single Dirichlet from a Kume Walker mixture.

    Attributes:
        l_values: The list of l_i values (irrespective of order since all a_i's
            are equal) associated with this Dirichlet distribution.
        a_i: The difference between the largest eigenvalue (\lambda_0) and all
            others (\lambda_i).  Assumes that the a_i's are all equal.
    """
    __slots__ = ['l_values', 'a_i', 'p']

    def __init__(self, l_values, a_i, p):
        self.l_values = l_values
        self.a_i = a_i
        self.p = p

    def weight(self):
        """Computes the weight associated with this Dirichlet."""
        l_values = self.l_values
        a_i = self.a_i
        p = self.p
        lgamma = math.lgamma

        # log_w accumulates the value of w for l_values.  It's initialized to
        # the log of the base term (outside the product) in w(l).
        log_w = lgamma(0.5) - lgamma(0.5 * (p + 1) + sum(l_values))

        # log_count accumulates the number of assignments of l that are based
        # on l_values and share the same value of w.
        log_count = 0.0

        # Note that partitions(0) returns [0] (for better or worse).
        if l_values[0] == 0:
            part_size = 0
        else:
            last_l = None
            multiplicity = 0
            for i, l_i in enumerate(l_values):
                if last_l == l_i:
                    multiplicity += 1
                else:
                    # Account for all of the permutations which are identical
                    # assignments of last_l.
                    log_count -= lgamma(multiplicity + 1)
                    last_l = l_i
                    multiplicity = 1

                # The log of the term of w(l) corresponding to a single i
                # (within the product).
                log_w += (l_i * math.log(a_i) + lgamma(l_i + 0.5)
                        - lgamma(l_i + 1))

            # Account for all of the permutations which are identical
            # assignments of the final last_l.
            log_count -= lgamma(multiplicity + 1)

            part_size = i + 1

        # Add in all of the 0 terms.
        log_w += (p - part_size) * (lgamma(0.5) - lgamma(+ 1))

        # The partial permutations function indicates the number of identical
        # terms (i.e., the number of l vectors that can be formed by
        # rearranging l_values).
        log_count += (lgamma(p + 1) - lgamma(p - part_size + 1))
        print(math.exp(log_w + log_count), l_values)
        return math.exp(log_w + log_count)

    def sample(self, rand):
        """Sample from the Bingham assuming this Dirichlet was picked."""
        l_array = list(self.l_values)
        rand.shuffle(l_array)

        alphas = [0.5] + [(l_i + 0.5) for l_i in l_array]
        gammas = [rand.gammavariate(alpha_i, 1) for alpha_i in alphas]
        total = sum(gammas)

        bingham = numpy.empty(self.p + 1)
        for i, gamma_i in enumerate(gammas):
            s_i = gamma_i / total
            sign = rand.choice([-1, 1])
            bingham[i] = sign * (s_i ** 0.5)

        return bingham


def kume_walker_norm2(a_i, p, N):
    """Sum up all of the w(l) terms with \sum l_i < N.

    This variant assumes that a_i = a_1 for all i.  Note that p is the number
    of dimensions.
    """
    total = 0
    for sum_l in range(N):
        for part in partitions(sum_l):
            if len(part) > p:
                continue

            d = KumeWalkerDirichlet(part, a_i, p)
            total += d.weight()

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
