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
from kdtree import KDTree

from cubes.cube import Cube

#------------------------------------------------------------------------------
# The simulation class

class Simulation(ParamObj):
    _params = dict(
        maximize=Param(default=0, doc='If 1, maximize instead of minimize'),
        )

class Simulation(VarArgs):
    _args = [
        ('wrap', False, 'Particles wrap around the constraints when True'),
        ('initspace', 1.0, 'Size of initialization space (per dimension)'),
        ('initoffset', 0.0, 'Offset of initialization space (per dimension)'),
        ('kmeaniters', 0, 'K-Means initialization iterations'),
        ('kdtreeinit', False, 'Use a k-d tree to reset or init new particles'),
        ('kdminsize', 0.01, 'Minimum fraction of feasible space for a cell'),
        ('kdusefitness', True, 'Balance fitness with volume'),
        ('kdrebuild', 0, 'Rebuild every so many evaluations'),
        ('kdrandvel', False, 'Use a random velocity when reinitializing'),
        ('kdaddonlynew', False, 'Add only the relocated particles to the tree'),
        ]

    def __init__( self,
            nparts, neighborhood, function, motion, *args, **kargs
            ):
        super(Simulation,self).__init__( *args, **kargs )

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

        ispace = self.initspace
        ioffset = self.initoffset

        constraints = [
            (cl+abs(cr-cl)*ioffset,cl + abs(cr-cl)*ioffset + (cr-cl)*ispace)
            for cl,cr in self.func.constraints
            ]
        sizes = [abs(cr-cl) * ispace for cl, cr in self.func.constraints]
        vconstraints = [(-s,s) for s in sizes]

        self.cube = Cube( constraints )
        self.vcube = Cube( vconstraints )

        corner1, corner2 = zip( *constraints )
        self.diaglen = abs(Vector(corner1) - Vector(corner2))

        self.kdminsizes = None
        if self.kdminsize > 0.0:
            self.kdminsizes = [s*self.kdminsize for s in sizes]

    def best_leaf( self ):
        """Returns the best leaf in the tree for particle initialization"""
        if self.kdusefitness:
            leaves = list(self.tree.iterleaves())
            tot = len(leaves)
            if tot <= 1:
                return self.tree.root

            volumes = [(l.volume(),i) for i, l in enumerate(leaves)]
            fitnesses = [(l.point.bestval,i) for i, l in enumerate(leaves)]

            mnvol = min(volumes)[0]
            mxvol = max(volumes)[0]
            mnfit = min(fitnesses)[0]
            mxfit = max(fitnesses)[0]

            if not self.maximize:
                mnfit, mxfit = mxfit, mnfit

            lenvol = mxvol - mnvol
            lenfit = mxfit - mnfit

            # Now compute a combined probability for each index
            probs = [1.0] * tot
            for fit, ileaf in fitnesses:
                try:
                    probs[ileaf] *= (fit - mnfit) / lenfit
                except ZeroDivisionError, e:
                    # No change to the probability, since this one is
                    # meaningless
                    pass

            for i, (vol, ileaf) in enumerate(volumes):
                try:
                    probs[ileaf] *= (vol - mnvol) / lenvol
                except ZeroDivisionError, e:
                    # No change to the probability, since this one is
                    # meaningless
                    pass

            # Finally, find the higest probability.
            piter = enumerate(probs)
            bestidx, bestprob = piter.next()
            for i, p in piter:
                if p > bestprob:
                    bestidx = i
                    bestprob = p

            return leaves[bestidx]

        else:
            # This approach just gets the biggest one
            bestvol = 0
            bestnode = None
            for node in self.tree.iterleaves():
                vol = node.volume()
                if bestvol < vol:
                    bestvol = vol
                    bestnode = node
            return bestnode

    def node_cubes( self, node ):
        c = Cube(node.constraints)
        sizes = [abs(cr-cl) for cl,cr in node.constraints]
        vc = Cube([(-s,s) for s in sizes])
        return c, vc

    def newparticle( self, **args ):
        args['numparts'] = 1
        piter = self.iternewparticles( **args )
        return piter.next()

    def iternewparticles( self, **args ):
        func = self.func
        for pos, vel in self.iternewstates( **args ):
            val = func( pos )
            yield Particle( pos, vel, val )

    def update_particle( self, particle, newpos, newvel ):
        if self.wrap:
            self.cube.wrap(particle.pos)
        particle.update( newpos, newvel, self.func(newpos), self.comparator )

    def reset_particle( self, particle, newpos, newvel, newval ):
        particle.reset( newpos, newvel, newval )

    def iterevals( self ):
        comp = self.comparator
        kargs = self.keywordargs
        self.neighborhood.setup(self, self.nparts, comparator=comp, **kargs)

        constraints = self.cube.constraints

        self.tree = KDTree( constraints )

        for i, (particle, soc) in enumerate(self.neighborhood):
            if self.kdtreeinit:
                if self.kdrebuild > 0 and i % self.kdrebuild == 0:
                    points = list(self.tree.iterpoints())
                    self.tree = KDTree.frompoints( constraints, points )
                if not self.kdaddonlynew or (hasattr(particle,'new') and particle.new):
                    self.tree.add_point( particle.copy(), minsizes=self.kdminsizes )
            yield soc, i

    def iterbatches( self ):
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
