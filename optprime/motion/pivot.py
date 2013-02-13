from __future__ import division
import basic
from itertools import izip

try:
    range = xrange
except NameError:
    pass


class Pivot(basic._Base):
    def hyperspherevariate(self, radius, uniformity=1):
        """Based on Clerc's sampling code.

        If uniformity < 0, returns a gaussian distribution with -uniformity
        as sdev.  If uniformity = 1, samples uniformly from a hypersphere.
        """
        # based on Clerc's code:
        pw = 1/self.dims
        dims = self.dims

        # ---- Direction ----
        vec = array([rand.gauss(0,1) for i in range(dims)])

        # ---- Random Radius ----
        if uniformity > 0:
            r = rand.uniform(0,1) ** (pw*uniformity)
        else:
            sdev = abs(uniformity)
            r = abs(rand.gauss(0,sdev))

        # ---- finish the vector ----
        frac = radius * r / abs(vec)
        return vec * frac

    def __call__(self, particle, rand):
        """Implements the Simple Pivot motion approach"""
        distance = abs(particle.pbestpos - particle.nbestpos)

        ppart = particle.pbestpos + self.hyperspherevariate( distance )
        pbest = particle.nbestpos + self.hyperspherevariate( distance )

        totval = particle.nbestval + particle.pbestval

        try:
            pw = particle.nbestval / totval
        except ZeroDivisionError, e:
            pw = 0.5

        # If we are maximizing rather than minimizing, we need to reverse the
        # weights.
        if self.comparator(1, 0):
            pw = 1-pw

        newpos = pw * ppart + (1-pw) * pbest
        newvel = newpos - particle.pos

        return newpos, newvel
