"""The main part of the PSO code.

Chris said that this was in the process of getting merged with neighborhood.
"""

from __future__ import division
from varargs import VarArgs
from random import Random
from particle import Particle
from operator import lt, gt
from itertools import izip, count
from Vector import Vector
from kdtree import KDTree

from cubes.cube import Cube

#------------------------------------------------------------------------------
# Set up the known neighborhoods
from neighborhood.fixed import Star, Ring, Wheel
from neighborhood.tribes import TRIBES
neighborhoodlist = [
    Star,
    Ring,
    Wheel,
    TRIBES,
]
neighborhoods = dict([(v.__name__, v) for v in neighborhoodlist])

#------------------------------------------------------------------------------
# Set up the known motion types
from motion.basic import Basic, BasicGauss, BasicAdaptive
from motion.bare import Bare
from motion.link import Link1, Link2, Link3, Link4
from motion.pivot import Pivot
motionlist = [
    Basic,
    BasicGauss,
    BasicAdaptive,
    Bare,
    Pivot,
    Link1,
    Link2,
    Link3,
    Link4,
]
# Some motions require scipy, Numeric, etc., and might not work everywhere.
try:
    from motion.kalman import Kalman
    motionlist.append(Kalman)
except ImportError:
    pass
motions = dict([(v.__name__, v) for v in motionlist])

#------------------------------------------------------------------------------
# Set up the known functions (only useful benchmarks, please)
from functions.ackley import Ackley
from functions.dejong import DeJongF4
from functions.griewank import Griewank
from functions.easom import Easom
from functions.rastrigin import Rastrigin
from functions.rosenbrock import Rosenbrock
from functions.schaffer import SchafferF6, SchafferF7
from functions.sphere import Sphere
from functions.monson import TwoGaussians, ValleyNeedle, AsymmetricCone
from functions.quadratic import Quadratic
from functions.distance import Distance
from functions.schwefel import Schwefel
from functions.pat import Pat
from functions.keane import Keane
from functions.gauss import Gauss
from functions.financial import AlphaBeta
from functions.rbf import RBF
from functions.butterfly import Butterfly
from functions.psocryst import Crystal
#from functions.art import Art
functionlist = [
    Sphere,
    Distance,
    Quadratic,
    DeJongF4,
    Rosenbrock,
    Rastrigin,
    Griewank,
    Ackley,
    SchafferF6,
    SchafferF7,
    TwoGaussians,
    ValleyNeedle,
    AsymmetricCone,
    Schwefel,
    Easom,
    Pat,
    Keane,
    Gauss,
    AlphaBeta,
    RBF,
    Butterfly,
    Crystal
#    ,Art
]
# Some functions require scipy, Numeric, etc., and might not work everywhere.
try:
    from functions.ackley import Ackley as x
    from functions.constrained10d import ConstrainedSphere10d, \
            ConstrainedRastrigin10d, ConstrainedRosenbrock10d, \
            ConstrainedGriewank10d, ConstrainedQuadratic10d
    functionlist += [
        ConstrainedSphere10d,
        ConstrainedQuadratic10d,
        ConstrainedRastrigin10d,
        ConstrainedRosenbrock10d,
        ConstrainedGriewank10d,
    ]
except ImportError:
    pass
functions = dict([(v.__name__, v) for v in functionlist])

#------------------------------------------------------------------------------
# The simulation class

class Simulation(VarArgs):
    _args = [
        ('maximize',False,'If true, maximizes the function'),
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
            nparts, neighborcls, funcls, motioncls, *args, **kargs
            ):
        super(Simulation,self).__init__( *args, **kargs )

        # Save these for later.
        self.keywordargs = kargs

        self.rand = kargs.get('rand', Random())

        self.dims = kargs['dims']
        self.nparts = nparts
        self.func = func = funcls(**kargs)
        self.neighborcls = neighborcls
        self.comparator = self.maximize and gt or lt
        self.motion = motioncls(
                self.comparator,
                self.func.constraints,
                **kargs
                )

        ispace = self.initspace
        ioffset = self.initoffset

        constraints = [
            (cl+abs(cr-cl)*ioffset,cl + abs(cr-cl)*ioffset + (cr-cl)*ispace)
            for cl,cr in func.constraints
            ]
        sizes = [abs(cr-cl) * ispace for cl, cr in func.constraints]
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

    def iternewstates( self, **args ):
        """Iterator that successively returns new position/velocity pairs
        within a possibly given set of constraints.
        """
        # Create the correct number of new particles
        positions = []
        rand = self.rand

        randvels = True

        # We allow the number of particles to be passed in, as well as the
        # number of kmeans iterations, but they are not necessary.  Note that
        # we don't just use the 'get' facility here to do this, since None
        # should *also* indicate that we didn't pass it in.
        numparts = args.get('numparts',None)
        kmeaniters = args.get('kmeaniters',None)

        if numparts is None:
            numparts = self.nparts

        if kmeaniters is None:
            kmeaniters = self.kmeaniters

        dims = self.dims

        constraints = args.get('constraints',None)
        if constraints is not None:
            # We also allow for passing in constraints
            sizes = [abs(cr-cl) for cl,cr in constraints]
            vconstraints = [(-s,s) for s in sizes]
            c = Cube(constraints)
            vc = Cube(vconstraints)
        else:
            # No constraints sent in?  Use the default.
            if self.kdtreeinit:
                leaf = self.best_leaf()
                c, vc = self.node_cubes( leaf )
                # Using the kdtree -- so we need to be sure to use the
                # appropriate preference.
                randvels = self.kdrandvel
            else:
                c = self.cube
                vc = self.vcube

        if kmeaniters:
            # Create initial mean positions
            means = [(Vector(c.random_vec(rand)),1) for i in xrange(numparts)]

            for i in xrange(kmeaniters * numparts):
                rvec = Vector(c.random_vec(rand))

                # Find the closest one:
                m_iter = enumerate(means)
                bestidx, (bestmean, bestnum) = m_iter.next()
                bestdist = abs(rvec - bestmean)
                for idx, (mean, num) in m_iter:
                    d = abs(rvec - mean)
                    if d < bestdist:
                        bestdist = d
                        bestmean = mean
                        bestnum = num
                        bestidx = idx
                
                # Now that we know who is closest, we can update the mean
                newmean = (bestnum * bestmean + rvec) / (bestnum + 1)
                means[bestidx] = (newmean, bestnum+1)

            positions = [pos for pos, num in means]

        for x in xrange(numparts):
            if not positions:
                newpos = c.random_vec(rand)
                if randvels:
                    newvel = vc.random_vec(rand)
                else:
                    newvel = Vector([0] * dims)
            else:
                newpos = positions[x]
                newvel = Vector([0] * dims)

            yield newpos, newvel

    def update_particle( self, particle, newpos, newvel ):
        if self.wrap:
            self.cube.wrap(particle.pos)
        particle.update( newpos, newvel, self.func(newpos), self.comparator )

    def reset_particle( self, particle, newpos, newvel, newval ):
        particle.reset( newpos, newvel, newval )

    def iterevals( self ):
        comp = self.comparator
        kargs = self.keywordargs
        soc = self.neighborcls( self, self.nparts, comparator=comp, **kargs )

        constraints = self.cube.constraints

        self.tree = KDTree( constraints )

        for i, (particle, soc) in enumerate(soc):
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
