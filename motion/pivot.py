from __future__ import division
import basic
from amlpso.Vector import Vector
from itertools import izip

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
        vec = Vector()
        for i in xrange(dims):
            vec.append(self.rand.gauss(0,1))

        # ---- Random Radius ----
        if uniformity > 0:
            r = self.rand.uniform(0,1) ** (pw*uniformity)
        else:
            sdev = abs(uniformity)
            r = abs(self.rand.gauss(0,sdev))

        # ---- finish the vector ----
        frac = radius * r / abs(vec)
        return vec * frac

    def __call__(self, particle):
        """Implements the Simple Pivot motion approach"""
        distance = abs(particle.bestpos - particle.gbestpos)

        ppart = particle.bestpos + self.hyperspherevariate( distance )
        pbest = particle.gbestpos + self.hyperspherevariate( distance )

        totval = particle.gbestval + particle.bestval

        try:
            pw = particle.gbestval / totval
        except ZeroDivisionError, e:
            pw = 0.5

        # If we are maximizing rather than minimizing, we need to reverse the
        # weights.
        if self.comparator(1, 0):
            pw = 1-pw

        newpos = pw * ppart + (1-pw) * pbest
        newvel = newpos - particle.pos

        return newpos, newvel
