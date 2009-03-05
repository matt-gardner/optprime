from __future__ import division
import basic
from amlpso.Vector import Vector
from itertools import izip

class Bare(basic._Base):
    def __call__(self, particle):
        """Implements the Bare Bones motion approach"""
        # Simply take the midpoint of the two as the mean and calculate the
        # standard deviation as the absolue value of the distance between them
        # in each dimension.
        mean = (particle.pbestpos + particle.nbestpos) / 2
        sdev = Vector([abs(x) for x in (particle.pbestpos - particle.nbestpos)])

        newpos = Vector([self.rand.gauss(m,s) for m,s in izip(mean,sdev)])
        newvel = newpos - particle.pos

        return newpos, newvel
