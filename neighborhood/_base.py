from __future__ import division

from aml.opt.varargs import VarArgs
from itertools import izip
import operator
from sets import Set
from aml.opt.Vector import Vector

class _Base(VarArgs):
    _args = [
        ('stagcount', 0,
        'Number of stagnations to trigger a relocation test (0 means no test)'),
        ('stagsize', 0.0,
        'Fraction of swarm that triggers relocation when tested'),
        ('bounceradius', 0.0,
        'Distance to other particles (relative to longest diagonal) '
        'that causes a bounce, 0 means no bounce'),
        ('bounceadapt', 1.0,
        'Use an adaptive bounce radius, one that decreases with every bounce'),
        ('bouncetournament', False,
        'The biggest particle in a bounce competition does not move'),
        ('bouncedistance', False,
        'Alter the bounce distance based on the adaptive parameter'),
        ('relocatebest', False, 'Always move the best differently',),
        ]
    def __init__( self, simulation, numparticles, *args, **kargs ):
        """Creates a sociometry object, which maintains particles and
        relationships.

        arguments:
            simulation -- the simulation governing this sociometry
            numparticles -- the number of initial particles to use
        keyword arguments:
            comparator -- defaults to operator.lt (minimization)
        """
        super(_Base,self).__init__( *args, **kargs )

        self.simulation = simulation
        self.rand = self.simulation.rand
        self.startparticles = numparticles

        self.is_better = kargs.get( 'comparator', operator.lt )

        # We create a nice simple id structure for the particles, where they
        # just have the id of their position in a list.
        self.particles = []

        # This tells us when all of the particles have been created.
        self.is_initialized = False
        # This tells us when we just emitted the first particle of a batch of
        # updates.
        self.start_of_batch = False
        # This tells us when we just emitted the last particle out of a batch
        # of updates.
        self.end_of_batch = True

        self.evaliter = self._iterevals()
        self.checkstagnation = self.stagcount > 0
        self.iters = 0
        self._bestparticle = None
        self.last_updated_particle = None

        self.trueradius = simulation.diaglen * self.bounceradius
        self.particleradii = {}

    # ------------------------------
    # Iterator protocol

    def __iter__( self ):
        return self

    def next( self ):
        p, soc = self.evaliter.next()
        self.last_updated_particle = p
        return p, soc

    # ------------------------------

    def _iterevals( self ):
        # First create the particles:
        n = self.startparticles
        sim = self.simulation
        self.start_of_batch = True
        self.end_of_batch = False
        for i, p in enumerate(sim.iternewparticles( numparts=n )):
            # Set up the index and yield self, from which everything can be
            # obtained.  We yield here because each particle creation is a
            # function evaluation.
            p.idx = i
            if self._bestparticle is None:
                self._bestparticle = p
            elif self.is_better(p.bestval, self._bestparticle.bestval):
                self._bestparticle = p
            self.addparticle( p )
            # We have to set initialized BEFORE we yield so that it's ready
            # AFTER we yield
            if i == n - 1:
                self.is_initialized = True
                self.end_of_batch = True
            self.iters += 1
            yield p, self
            self.start_of_batch = False

        # Start the regular evaluation cycle
        stagcheck = self.checkstagnation
        stagcount = self.stagcount
        stagthresh = self.stagsize
        while True:
            # Now update all of the particles in batch
            np = self.numparticles()
            newstates = [None] * np
            relocate = Set()
            # We move them in batches, which requires us to keep track of a set
            # of new values and to set them all at once.
            particles = tuple(self.iterparticles())
            bestpart = None
            bestval = None
            bestidx = None

            sim.motion.pre_batch( self )
            for i, p in enumerate(particles):
                if not hasattr(p, 'radius'):
                    p.radius = self.trueradius
                if stagcheck and p.stagnantcount >= stagcount:
                    relocate.add(i)
                if bestpart is None or self.is_better(p.bestval,bestval):
                    bestpart = p
                    bestval = p.bestval
                    bestidx = i
                best = self.bestneighbor( p )
                newstates[i] = sim.motion( p, best )
            sim.motion.post_batch( self )
            bestpos = bestpart.bestpos

            if self.relocatebest:
                relocate.add(bestidx)
            else:
                # We practice elitism so that we don't lose the best ever
                # position
                relocate.discard(bestidx)

            numrelocate = len(relocate)
            isstagnant = stagcheck and numrelocate >= int(stagthresh * np)
            if isstagnant or self.relocatebest:
                # Now we relocate all of the bad particles.  These will also be
                # marked in the newstates array so that they are not moved
                # again.
                stateiter = sim.iternewstates( numparts = numrelocate )
                for i, state in izip(relocate, stateiter):
                    newstates[i] = state
            else:
                # Empty out our relocate set so that we don't use it later.
                relocate = Set()

            if self.trueradius > 0.0:
                # Are we going to do the spatial extension thing?  If so, we
                # need to check for collisions.
                to_bounce = Set()
                to_adapt = Set()
                # Search for collisions
                for i in xrange(np-1):
                    if newstates[i][0] is None:
                        posi = particles[i] + newstates[i][1]
                    else:
                        posi = Vector(newstates[i][0])
                    ri = particles[i].radius
                    for j in xrange(i+1, np):
                        if newstates[j][0] is None:
                            posj = particles[j] + newstates[j][1]
                        else:
                            posj = Vector(newstates[j][0])
                        rj = particles[j].radius
                        if abs(posj - posi) <= (ri + rj):
                            # Bounce these
                            to_adapt.add( i )
                            to_adapt.add( j )
                            if self.bouncetournament and ri != rj:
                                # Only bounce the smaller particle out.  If
                                # equal, bounce both
                                if rj < ri:
                                    to_bounce.add( j )
                                else:
                                    to_bounce.add( i )
                            else:
                                to_bounce.add( i )
                                to_bounce.add( j )
                # Now do the actual bouncing
                for i in to_bounce:
                    p = particles[i]
                    pos, vel = newstates[i]
                    if pos is None:
                        pos = p + vel
                    scale = 1.0
                    if self.bouncedistance:
                        scale *= self.trueradius / p.radius
                    newstates[i] = (p - scale * (pos - p), -vel)

                for i in to_adapt:
                    particles[i].radius *= self.bounceadapt

            # Here we set them and yield with each movement, since it
            # represents a function evaluation.  Note that we iterate over a
            # copy in case the number of particles changes just after the final
            # yield (but before the for loop is evaluated again)
            particles = tuple(self.iterparticles())
            n = len(particles)
            self.start_of_batch = True
            self.end_of_batch = False
            for i, (p, state) in enumerate(izip(particles, newstates)):
                # This is first because we want to know if this is the last
                # particle before the update or reset functions are actually
                # called.
                if i == n - 1:
                    self.end_of_batch = True
                if i in relocate:
                    pos, vel = state
                    val = sim.func(pos)
                    self.simulation.reset_particle( p, pos, vel, val )
                    if i == bestidx:
                        p.bestval = bestval
                        p.bestpos = bestpos
                    p.new = True
                else:
                    self.simulation.update_particle( p, *state )
                    if relocate:
                        p.resetcounts()
                    p.new = False
                self.iters += 1
                if self.is_better( p.bestval, self._bestparticle.bestval ):
                    self._bestparticle = p
                yield p, self
                self.start_of_batch = False

    def addparticle( self, particle ):
        self.particles.append( particle )

    def iterparticles( self ):
        return iter(self.particles)

    def iterneighbors( self, particle ):
        if not self.is_initialized:
            raise ValueError( 'Swarm is not yet initialized!' )
        if self.__class__ is _Base:
            raise NotImplementedError( 'iterneighbors' )

    def bestparticle( self ):
        return self._bestparticle

    def bestneighbor( self, particle ):
        """Returns the best neighbor for this particle"""
        niter = self.iterneighbors( particle )
        try:
            best = niter.next()
        except StopIteration:
            return None

        for p in niter:
            if self.is_better( p.bestval, best.bestval ):
                best = p
        return best

    def numparticles( self ):
        return len(self.particles)


