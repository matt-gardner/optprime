from __future__ import division
from math import exp
from . import Benchmark

try:
    range = xrange
except NameError:
    pass


class Quadratic(Benchmark):
    _each_constraints = (-100, 100)

    def _standard_call(self, vec):
        s = 0
        for i in range(self.dims):
            for j in range(self.dims):
                s += exp(-((vec[i] - vec[j])**2)) * vec[i] * vec[j]
        return s + sum(vec[:self.dims])
