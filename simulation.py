"""The main part of the PSO code.

Chris said that this was in the process of getting merged with neighborhood.
"""

from __future__ import division
from mrs.param import ParamObj, Param
from random import Random
from particle import Particle
from operator import lt, gt
from itertools import izip, count
from Vector import Vector

from cubes.cube import Cube

#------------------------------------------------------------------------------
# The simulation class

class Simulation(ParamObj):
    _params = dict(
        maximize=Param(default=0, doc='If 1, maximize instead of minimize'),
        )

    def __init__( self,
            nparts, neighborhood, function, motion, *args, **kargs
            ):
        super(Simulation,self).__init__(*args, **kargs)

        # Save these for later.
        self.keywordargs = kargs

        self.rand = kargs.get('rand', Random())

        self.dims = kargs['dims']
        self.nparts = nparts
        self.func = function
        self.neighborhood = neighborhood
        self.motion = motion
        self.comparator = self.maximize and gt or lt

        self.func.setup(self.dims)
        self.motion.setup(self.comparator, self.func.constraints)

    def iterevals(self):
        comp = self.comparator
        kargs = self.keywordargs
        self.neighborhood.setup(self, self.nparts, comparator=comp, **kargs)

        constraints = self.cube.constraints

        for i, (particle, soc) in enumerate(self.neighborhood):
            yield soc, i

    def iterbatches(self):
        eiter = self.iterevals()
        iters = 0
        while True:
            soc, i = eiter.next()
            if soc.is_initialized:
                break
        iters += 1
        yield soc, iters

        # Now that we're initialized:
        while True:
            for i in xrange(soc.numparticles()):
                soc, i = eiter.next()
            iters += 1
            yield soc, iters
