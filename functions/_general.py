from __future__ import division

from itertools import izip
import operator

from mrs.param import ParamObj, Param

class _Base(ParamObj):
    _params = dict(
            dims=Param(default=2, type='int', shortopt='-d',
                doc='Number of dimensions'),
            # center is a string because it might be multidimensional
            center=Param(default='', type='str',
                doc='Relative center, between 0 and 1 per dim (rand if '')'),
            maximize=Param(type='bool',
                doc='Maximize the function instead of minimizing.'),
            success=Param(default=10.0**(-10), type='float',
                doc='Success value (the algorithm stops when reached)'),
            )

    def master_log(self):
        """print something after the master's logs"""
        pass

    def setup(self):
        self._set_constraints(((0,0),)*self.dims)
        if self.maximize:
            self.comparator = operator.gt
        else:
            self.comparator = operator.lt

    def randomize_center(self, rand):
        """Initializes a random center using the given generator.

        Note that if a specific function center was given as a command-line
        parameter, then the center will _not_ be reset.
        """
        if not self.center:
            center = [rand.uniform(0, 1) for i in xrange(self.dims)]
            self._set_abscenter(center)

    def _set_constraints(self, constraints):
        self.dims = len(constraints)
        self.constraints = tuple(constraints)
        if ',' in self.center:
            center = [float(x) for x in self.center.split(',')]
        else:
            if self.center:
                val = float(self.center)
            else:
                val = 0.0
            center = [val] * self.dims
        self._set_abscenter(center)

    def _set_abscenter(self, center):
        """Sets an absolute center from the relative center and constraints."""
        self.abscenter = [
            (c * (r - l) + l)
            for c, (l, r) in izip(center, self.constraints)]

    def __call__(self, vec):
        return 0

    def is_opt(self, value):
        """Determines whether the value is officially optimal.

        In other words, whether the value is sufficiently low/high.
        """
        return self.comparator(value, self.success)
