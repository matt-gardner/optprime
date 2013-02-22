from __future__ import division

from math import sqrt, sin
from ._base import Benchmark


class SchafferF6(Benchmark):
    _each_constraints = (-100, 100)

    def _standard_call(self, vec):
        xsq = sum(imap(lambda x: x**2, vec))
        return 0.5 + (sin(sqrt(xsq))**2 - 0.5)/((1 + 0.001*xsq)**2)


class SchafferF7(Benchmark):
    _each_constraints = (-100, 100)

    def _standard_call(self, vec):
        xsq = sum(imap(lambda x: x**2, vec))
        return xsq**0.25 * (sin(50*xsq**0.1)**2 + 1)
