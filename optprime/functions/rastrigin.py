from __future__ import division

from math import cos, pi
from . import Benchmark

try:
    from itertools import izip as zip
except ImportError:
    pass


class Rastrigin(Benchmark):
    _each_constraints = (-5.12, 5.12)

    def _standard_call(self, vec):
        s = 0.0
        for v in vec:
            s += v**2 - 10*cos(2*pi*v) + 10
        return s
