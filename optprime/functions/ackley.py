from __future__ import division
from itertools import imap
from math import e, exp, sqrt, cos, pi

from ._base import Benchmark


class Ackley(Benchmark):
    _each_constraints = (-32.768, 32.768)

    def __call__(self, vec):
        n = self.dims
        # Sum the squares
        s1 = sum(imap(lambda x,c: (x-c)**2, vec, self.abscenter))
        # Add up the cosine thingy
        s2 = sum(imap(lambda x,c: cos(2*pi*(x-c)), vec, self.abscenter))
        return 20 + e + -20 * exp(-0.2 * sqrt(s1/n)) - exp(s2/n)
