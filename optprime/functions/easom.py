from __future__ import division
from math import exp, cos, pi
from operator import mul
from ._base import Benchmark

class Easom(Benchmark):
    _each_constraints = (-100, 100)

    # fEaso(x1,x2)=-cos(x1) cos(x2) exp(-((x1-pi)^2+(x2-pi)^2));
    # Modified to have the center at 0,0 instead of pi and minimum of 0:
    # fEaso(x1,x2)=1-cos(x1+pi) cos(x2+pi) exp(-(x1^2+x2^2));
    # To move the center simply subtract from each coordinate
    # -100<=x(i)<=100, i=1:2.
    def _standard_call(self, vec):
        n = self.dims
        # Sum the squares
        sqs = sum(imap(lambda x: (x-c)**2, vec))
        # Calculate the product of cosines
        cosines = reduce(mul, imap(lambda x: cos(x+pi-c), vec))
        # Complete the function:
        return 1 + exp(-sqs) * cosines
