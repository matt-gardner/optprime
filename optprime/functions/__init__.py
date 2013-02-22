from __future__ import division

import operator

from mrs.param import ParamObj, Param

try:
    from numpy import array, empty
except ImportError:
    from numpypy import array, empty

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
            )

    _each_constraints = (0, 0)

    def setup(self, rand):
        # Set the constraints (feasible region).
        self.constraints = array([self._each_constraints] * self.dims)

        # Set the function center for shifting (may be later overridden by
        # randomize_center).  Note that self.center is the user-specified
        # value, and self.abscenter is what actually gets used.
        center = empty(self.dims)
        if ',' in self.center:
            center[:] = [float(x) for x in self.center.split(',')]
        elif self.center:
            center[:] = float(self.center)
        else:
            center = array([rand.uniform(0, 1) for i in range(self.dims)])
        left = self.constraints[:, 0]
        right = self.constraints[:, 1]
        self.abscenter = center * (right - left) + left

        # Setup maximization vs. minimization.
        if self.maximize:
            self.comparator = operator.gt
        else:
            self.comparator = operator.lt

    def __call__(self, vec):
        return self._standard_call(vec - self.abscenter)

    def _standard_call(self, vec):
        """Evaluate the function in its standard form.

        Assumes that any shifting, scaling, etc., are done externally.  This
        method should be overridden.
        """
        return 0
