from __future__ import division

import operator

from mrs.param import ParamObj, Param
from ..linalg import rand_cliques_matrix, rand_perm_matrix

try:
    import numpy as np
except ImportError:
    import numpypy as np

try:
    from itertools import izip as zip
except ImportError:
    pass

try:
    range = xrange
except NameError:
    pass


class BaseFunction(ParamObj):
    """An arbitrary objective function."""
    _params = dict(
            success=Param(default=10.0**(-10), type='float',
                doc='Success value (the algorithm stops when reached)'),
            )

    def master_log(self):
        """print something after the master's logs"""
        pass

    def setup(self, rand):
        self.dims = 0
        self.comparator = operator.lt

    def __call__(self, vec):
        return 0

    def is_opt(self, value):
        """Determines whether the value is officially optimal.

        In other words, whether the value is sufficiently low/high.
        """
        return self.comparator(value, self.success)


class Benchmark(BaseFunction):
    """An objective function that is a tweakable benchmark function.

    Features include: randomized centers, configurable numbers of dimensions,
    etc.
    """
    _params = dict(
            dims=Param(default=2, type='int', shortopt='-d',
                doc='Number of dimensions'),
            # center is a string because it might be multidimensional
            center=Param(default='', type='str',
                doc='Relative center, between 0 and 1 per dim (rand if '')'),
            maximize=Param(type='bool',
                doc='Maximize the function instead of minimizing.'),
            part_sep_cliques=Param(default=0, type='int',
                doc='Number of partially separable cliques of unsep. variables'
                    ' (0 for fully sep., 1 for fully unsep.)'),
            )

    _each_constraints = (0, 0)

    def setup(self, rand):
        # Set the constraints (feasible region).
        self.constraints = np.array([self._each_constraints] * self.dims)

        # Set the function center for shifting (may be later overridden by
        # randomize_center).  Note that self.center is the user-specified
        # value, and self.abscenter is what actually gets used.
        center = np.empty(self.dims)
        if ',' in self.center:
            center[:] = [float(x) for x in self.center.split(',')]
        elif self.center:
            center[:] = float(self.center)
        else:
            center = np.array([rand.uniform(0, 1) for i in range(self.dims)])
        left = self.constraints[:, 0]
        right = self.constraints[:, 1]
        self.abscenter = center * (right - left) + left

        # Setup maximization vs. minimization.
        if self.maximize:
            self.comparator = operator.gt
        else:
            self.comparator = operator.lt

        # Setup partially separable cliques of unseparable variables.
        if self.part_sep_cliques:
            A = rand_cliques_matrix(self.dims, self.part_sep_cliques, rand)
            B = rand_perm_matrix(self.dims, rand)
            self._sep_matrix = np.dot(A, B)
        else:
            self._sep_matrix = None

    def __call__(self, vec):
        vec = vec - self.abscenter
        if self._sep_matrix is not None:
            rotated = np.dot(self._sep_matrix, vec)
            vec = np.squeeze(rotated)
        return self._standard_call(vec)

    def _standard_call(self, vec):
        """Evaluate the function in its standard form.

        Assumes that any shifting, scaling, etc., are done externally.  This
        method should be overridden.
        """
        return 0
