from __future__ import division
from itertools import izip, imap
from math import exp, e, sqrt, cos, pi
import _general

class Ackley(_general._Base):
    def setup(self):
        super(Ackley,self).setup()
        self._set_constraints(((-32.768,32.768),) * self.dims)

    def __call__( self, vec ):
        n = self.dims
        # Sum the squares
        s1 = sum(imap(lambda x,c: (x-c)**2, vec, self.abscenter))
        # Add up the cosine thingy
        s2 = sum(imap(lambda x,c: cos(2*pi*(x-c)), vec, self.abscenter))
        return 20 + e + -20 * exp(-0.2 * sqrt(s1/n)) - exp(s2/n)
