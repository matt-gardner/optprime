"""Miscellaneous linear algebra operations.

For numpy arrays vs. matrices, see:
    http://www.scipy.org/NumPy_for_Matlab_Users#head-e9a492daa18afcd86e84e07cd2824a9b1b651935
"""

from __future__ import division, print_function

import bisect
import collections
import itertools
import math
import random

try:
    import numpy as np
except ImportError:
    import numpypy as np

def eigh_sorted(A):
    """Performs numpy.linalg.eigh and sorts the resulting eigenvalues/vectors.

    Note that sorting is from large to small.
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    idx = np.argsort(eigvals[::-1])
    return eigvals[idx], eigvecs[:, idx]

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

def chol_update(L, x):
    """Rank 1 update of the lower-triangular Cholesky factor L.

    L is modified in-place.
    """
    x = x.copy()
    for k in range(len(x)):
        r = (L[k, k] ** 2 + x[k] ** 2) ** 0.5
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        L[k+1:, k] = (L[k+1:, k] + s * x[k+1:]) / c
        x[k+1:] = c * x[k+1:] - s * L[k+1:, k]

def chol_downdate(L, x):
    """Rank 1 downdate of the lower-triangular Cholesky factor L.

    L is modified in-place.
    """
    x = x.copy()
    for k in range(len(x)):
        r_squared = L[k, k] ** 2 - x[k] ** 2
        if r_squared <= 0.0:
            raise NonPosDefError()
        r = r_squared ** 0.5
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        L[k+1:, k] = (L[k+1:, k] - s * x[k+1:]) / c
        x[k+1:] = c * x[k+1:] - s * L[k+1:, k]

class NonPosDefError(Exception):
    pass

class BinghamSampler(object):
    """Sample from a Bingham distribution.

    The pdf is of the form:
        f(z) = c(A)^{-1} exp(z^T A z), z \in S^{k-1}

    where A is a k*k symmetric matrix, c(A) is a normalizing constant, and
    S^{k-1} is the unit sphere in R^k.

    Methods based on: Kent, Constable, and Er.  Simulation for the complex
    Bingham distribution.  Statistics and Computing, 2004.  We skip Method 3,
    which is only useful in very limited circumstances (all lambdas equal
    to about 0.5).

    Parameters (either A or lambdas must be specified, but not both):
        A: the parameter matrix of the Bingham distribution
        lambdas: the first k-1 eigenvalues of -A (the smallest is assumed to
            be 0 and is not included in the list).
    """
    def __init__(self, A=None, lambdas=None):
        assert lambdas is None or A is None

        self._sampler = None
        self._eigvecs = None

        if A is not None:
            eigvals, self._eigvecs = eigh_sorted(-A)
            smallest_eig = eigvals[-1]
            lambdas = eigvals[:-1] - smallest_eig
        self._lambdas = lambdas

    def _pick_sampler(self):
        if any(l == 0 for l in self._lambdas):
            return self.sample_m2

        k = len(self._lambdas) + 1

        # From Table 1: expected number for M1 with p_T removed.
        m1 = math.log(k - 1)
        for lambda_j in self._lambdas:
            m1 += math.log(1 - math.exp(-lambda_j))

        # From Table 1: expected number for M2 with p_T removed.
        m2 = math.log(k)
        for lambda_j in self._lambdas:
            m2 += math.log(lambda_j)
        m2 -= math.lgamma((k - 1) + 1)

        if m1 < m2:
            return self.sample_m1
        else:
            return self.sample_m2

    def sample(self, rand):
        if self._sampler is None:
            self._sampler = self._pick_sampler()
        return self._sampler(rand)

    def sample_m1(self, rand):
        """Sample using Method 1: Truncation to the simplex."""
        k = len(self._lambdas) + 1

        while True:
            uniforms = [rand.random() for _ in range(k - 1)]
            s = [-(1 / l_j) * math.log(1 - u_j * (1 - math.exp(-l_j)))
                for l_j, u_j in zip(self._lambdas, uniforms)]
            if sum(s) < 1:
                return self._convert_s_to_z(s)

    def sample_m2(self, rand):
        """Sample using Method 2: Acceptance-rejection on the simplex."""
        k = len(self._lambdas) + 1

        while True:
            uniforms = [rand.random() for _ in range(k - 1)]
            uniforms.sort()
            last = 0

            s = []
            for u in uniforms:
                s.append(u - last)
                last = u

            u = math.log(rand.random())
            if u < sum((-l_j * s_j) for l_j, s_j in zip(self._lambdas, s)):
                return self._convert_s_to_z(s)

    def _convert_s_to_z(self, s):
        """Convert a list of values on the simplex to values on the sphere."""
        s.append(1 - sum(s))
        s = np.array(s)
        z = s ** 0.5

        if self._eigvecs is not None:
            z = self._eigvecs.dot(z)
        return z

class BinghamWishartModel(object):
    """A Wishart random variable with Bingham-distributed observations.

    The distribution of A is a Wishart parameterized by its inverse-scale
    matrix.  The inverse-scale parameter is constrained such that it must be
    positive definite and the angle of its first eigenvector from the x-axis
    must be less than pi/4.

    The prior distribution of A is...

    Note that k has an extremely skewed distribution, such that it makes more
    sense to use the maximum likelihood estimate of k than to sample from it
    (the probability of the second most likely value is 5 to 6 orders of
    magnitude smaller in some quick experiments).

    The failure Bingham distribution's parameter matrix is the Wishart prior,
    while the success Bingham distribution's parameter matrix is the negative
    of the Wishart prior.

    Attributes:
        inv_scale_L: the lower-triangular component of the Cholesky
            decomposition of the inverse of the scale matrix of the Wishart
            distribution
        dof: the degrees of freedom of the Wishart distribution
    """
    def __init__(self, inv_scale_L, dof):
        m, n = inv_scale_L.shape
        assert m == n

        self._inv_scale_L = inv_scale_L
        self._dims = n
        self._dof = dof

    def incremented_dof(self, exp_scatter, n=1):
        """Create a new BinghamWishartModel with an incremented dof.

        The number of degrees of freedom is incremented by the given amount,
        and the prior scatter matrix is increased accordingly.
        """
        dof = self._dof + n
        old_inv_scale = np.dot(self._inv_scale_L, self._inv_scale_L.T)
        inv_scale = old_inv_scale + exp_scatter * n
        inv_scale_L = np.linalg.cholesky(inv_scale)
        return BinghamWishartModel(inv_scale_L, dof)

    def posterior_success(self, x):
        """Find the posterior given a sample from the success Bingham."""
        inv_scale_L = self._inv_scale_L.copy()
        dof = self._dof + 1

        chol_update(inv_scale_L, x)
        return BinghamWishartModel(inv_scale_L, dof)

    def posterior_failure(self, x):
        """Find the posterior given a sample from the success Bingham."""
        inv_scale_L = self._inv_scale_L.copy()
        dof = self._dof + 1

        chol_downdate(inv_scale_L, x)
        return BinghamWishartModel(inv_scale_L, dof)

    def sample_wishart(self, rand):
        """Sample from a Wishart with the given scale and degrees of freedom.

        The scale matrix must be a positive definite.  If the scale matrix has
        dimension p, then the degrees of freedom must satisfy dof > p-1.

        Based on: Smith and Hocking. Wishart Variate Generator. 1972.
        """
        # Note that it's slightly faster for large (larger than 40x40)
        # matrices to use this instead:
        #   scipy.linalg.solve_triangular(R, np.identity(len(R)))
        # or even better: scipy.linalg.get_lapack_funcs('trtri') or something
        scale_L = np.linalg.inv(self._inv_scale_L)
        m, n = scale_L.shape
        A = np.zeros((m, n))
        for i in range(m):
            # The Chi-squared distribution is a special case of the Gamma
            # distribution.  Note that Python uses the scale parameterization
            # and that the paper uses 1-based instead of 0-based indexing.
            A[i, i] = rand.gammavariate((self._dof - i) / 2, 2) ** 0.5
        for i in range(1, m):
            for j in range(i):
                A[i, j] = rand.normalvariate(0, 1)
        LA = np.dot(scale_L, A)
        return np.dot(LA, LA.T)

    def sample_success(self, rand):
        """Sample from the success Bingham distribution."""
        A = self.sample_wishart(rand)
        bs = BinghamSampler(-A/2)
        return bs.sample(rand)

def make_bingham_wishart_model(dims, kappa, rand):
    """Construct a new BinghamWishartModel with the given dimensions."""

    inv_scale = np.zeros((dims, dims))
    dof = 0

    exp_scatter = expected_mf_scatter(dims, kappa, rand)
    empty_model = BinghamWishartModel(inv_scale, dof)
    model = empty_model.incremented_dof(exp_scatter, dims)

    return model, exp_scatter

class UnobservedBinghamWishartModel(object):
    """Bingham-Wishart Model without fully observed success/failure data.
    """
    def __init__(self, bingham_wishart, p_s, p_t_1, p_t_0):
        self._bingham_wishart = bingham_wishart

        self._log_p_s = math.log(p_s)
        self._log_q_s = math.log(1 - p_s)
        self._log_p_t_1 = math.log(p_t_1)
        self._log_q_t_1 = math.log(1 - p_t_1)
        self._log_p_t_0 = math.log(p_t_0)
        self._log_q_t_0 = math.log(1 - p_t_0)

        # Outcomes of imperfect success/failure tests: (x, t) pairs.
        self._observations = []
        # Current guesses of latent success/failure state.
        self._successes = []
        # Current sample from Wishart distribution
        self._A_sample = None

    def sample_success(self, rand):
        return self._bingham_wishart.sample_success(rand)

    def record(self, x, t):
        """Record the outcome t of a test at direction x.

        Initially assumes that the latent state is `success`, since an invalid
        failure would cause inconsistency.  Thus, the `gibbs_successes`
        function should be called subsequently.
        """
        self._observations.append((x, t))
        self._successes.append(True)
        self._bingham_wishart = self._bingham_wishart.posterior_success(x)

    def gibbs_wishart(self, rand):
        """Sample from the Wishart distribution of node A."""
        self._A_sample = self._bingham_wishart.sample_wishart(rand)

    def gibbs_successes(self, rand, n=None):
        """Sample from the complete conditional of a latent success state node.

        If an observation number `n` is not given, a random one is chosen.
        """
        if n is None:
            n = rand.randrange(len(self._observations))
        x, t = self._observations[n]

        A = np.linalg.inv(self._bingham_wishart._inv_scale_L)
        xT_A_x = np.dot(x, np.dot(A, x))

        log_p = self._log_p_s - 0.5 * xT_A_x
        log_q = self._log_q_s + 0.5 * xT_A_x
        if t:
            log_p += self._log_p_t_1
            log_q += self._log_q_t_1
        else:
            log_p += self._log_p_t_0
            log_q += self._log_q_t_0

        p = math.exp(log_p)
        q = math.exp(log_q)

        success = (rand.uniform(0, p + q) < p)
        if success == self._successes[n]:
            return

        inv_scale_L = self._bingham_wishart._inv_scale_L.copy()
        dof = self._bingham_wishart._dof

        if success:
            chol_update(inv_scale_L, 2 * x)
        else:
            try:
                chol_downdate(inv_scale_L, 2 * x)
            except NonPosDefError:
                return

        self._bingham_wishart = BinghamWishartModel(inv_scale_L, dof)
        self._successes[n] = success

def sample_von_mises_fisher(dims, kappa, rand):
    """Samples from a von Mises Fisher distribution centered at [1, 0, ..., 0].

    To use a distribution at a different center, rotate the sample with a
    Householder transformation.
    """
    theta = rand.vonmisesvariate(0, kappa)
    head = np.array([math.cos(theta)])
    tail = math.sin(theta) * rand_norm_array(dims - 1, rand)
    return np.concatenate((head, tail))

def sample_mf_scatter(dims, kappa, num_samples, rand):
    """Sample from the scatter matrix of a set of von Mises Fisher samples.
    """
    samples = np.vstack([sample_von_mises_fisher(dims, kappa, rand)
            for _ in range(num_samples)])
    A = np.dot(samples.T, samples)
    return A

def expected_mf_scatter(dims, kappa, rand, samples=100000, step=10):
    """Find the expected value of the scatter matrix of a von Mises Fisher.

    The `rtol` parameter is the relative tolerance used in the stopping
    criterion.  The `step` parameter signifies how many von Mises Fisher
    samples to take at a time.

    Note that: Var(\sum{X_i} / n) = sigma^2 / n
        where sigma^2 is the variance of each independent X_i.
    So to divide the standard deviation by x, we need to multiply the number
    of samples by x^2.
    If the range of X_i is restricted to [0, 1] and its distribution is
    unimodal, then sigma^2 < 1/12.  This loose bound seems fairly effective
    at estimating the error of the diagonal entries in the scatter matrix.
    So samples=100000 gives almost four digits of accuracy.
    """
    assert samples % step == 0

    E = np.zeros((dims, dims))
    for _ in range(0, samples, step):
        A = sample_mf_scatter(dims, kappa, step, rand)

        E += A / samples

    # Average to reduce the total # of required samples.

    # Set indices for the diagonal (except the first entry).
    subdiag_rows, subdiag_cols = np.diag_indices(dims)
    subdiag_rows = subdiag_rows[1:]
    subdiag_cols = subdiag_cols[1:]

    # Set indices for the lower subtriangular entries (except the first col).
    subtril_rows, subtril_cols = np.tril_indices(dims - 2)
    subtril_rows = subtril_rows + 2
    subtril_cols = subtril_cols + 1
    subtriu_rows = dims - subtril_rows
    subtriu_cols = dims - subtril_cols

    # Average the first column (except the first entry).
    subcol1_mean = np.mean(E[1:, 0])
    E[1:, 0] = subcol1_mean
    E[0, 1:] = subcol1_mean
    # Average the diagonal (except the first entry).
    subdiag_mean = np.mean(E[subdiag_rows, subdiag_cols])
    E[subdiag_rows, subdiag_cols] = subdiag_mean
    # Average everything else.
    subtri_mean = np.mean(E[subtril_rows, subtril_cols])
    E[subtril_rows, subtril_cols] = subtri_mean
    E[subtriu_rows, subtriu_cols] = subtri_mean
    return E


# vim: et sw=4 sts=4
