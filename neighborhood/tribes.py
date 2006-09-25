from __future__ import division
from util import VarArgs
from _base import _Base

#------------------------------------------------------------------------------

class _Tribe(object):
    def __init__( self, soc, idx, *args, **kargs ):
        super(_Tribe,self).__init__( *args, **kargs )
        self.idx = idx
        self.soc = soc
        self.simulation = soc.simulation
        self.is_better = soc.simulation.comparator
        self.rand = self.soc.rand

        self.particles = {}

    def is_notworse( self, v1, v2 ):
        return v1 == v2 or self.is_better( v1, v2 )

    def add_particle( self, p ):
        self.particles[p.idx] = p
        p.tribe = self

    def del_particle( self, p ):
        del self.particles[p.idx]

    def is_good( self ):
        numgood = 0
        for p in self.particles.itervalues():
            if p.improvedcount >= 1:
                numgood += 1
        return self.rand.randrange(0,len(self.particles)) < numgood

    def extreme_particles( self ):
        piter = self.particles.itervalues()
        best = worst = piter.next()
        for p in piter:
            if self.is_better( p.bestval, best.bestval ):
                best = p
            if self.is_notworse( worst.bestval, p.bestval ):
                worst = p
        return best, worst

#------------------------------------------------------------------------------

class TRIBES(_Base):
    def __init__( self, *args, **kargs ):
        super(TRIBES,self).__init__( *args, **kargs )

        self.rand = self.simulation.rand
        self.curtribe = 0
        self.curparticle = 0
        self.tribes = {}
        self.particles = {}

    def _iterevals( self ):
        # We are overriding this function, but we really only want to intervene
        # in strategic locations, so we mostly leave the functionality intact.

        simpleiter = super(TRIBES,self)._iterevals()

        # Particle creation happens very naturally, but we want to intercede to
        # make the particles members of a tribe.  This is actually accomplished
        # in the addparticle method, which is called each time a particle is
        # created by the superclass.  We just make sure that the right
        # information gets there initially.

        # Therefore, addparticle checks for the presence of self.firsttribe to
        # do its initialization work, which can be deleted afterward.

        self.firsttribe = self.new_tribe()

        while not self.is_initialized:
            yield simpleiter.next()

        del self.firsttribe

        # We will need to create new particles with higher IDs than those
        # previously created.
        self.curparticle = len(self.particles)

        # Now we intercept every function evaluation.  When we have reached a
        # point where restructuring is to be done, we do it and carry on.

        steps_to_restructure = self.num_connections() // 2

        while True:
            # We only restructure in batches, after all particles have moved:
            for dummy in xrange(self.numparticles()):
                yield simpleiter.next()

            # Restructuring occurs after a number of batches have been
            # performed.  Since restructuring involves creating particles, we
            # yield for every particle created.
            steps_to_restructure -= 1
            if steps_to_restructure <= 0:
                steps_to_restructure = self.num_connections() // 2
                for p in self._restructure():
                    yield p, self

    def addparticle( self, particle ):
        """Overridden to ensure that the particle has appropriate structures in
        it and add it to the main list of particles.
        """
        particle.ext_informers = {}
        self.particles[particle.idx] = particle

        # We assume that there is a firsttribe variable is we aren't
        # initialized.
        if not self.is_initialized:
            self.firsttribe.add_particle( particle )

    def make_particle( self, tribe, constraints=None ):
        p = self.simulation.newparticle( constraints=constraints )

        p.idx = self.curparticle
        self.curparticle += 1
        self.addparticle( p )
        tribe.add_particle( p )
        return p

    def kill_particle( self, particle ):
        particle.tribe.del_particle( particle )
        del self.particles[particle.idx]

    def iterparticles( self ):
        return self.particles.itervalues()

    def iterneighbors( self, particle ):
        # All of the members of the tribe are informers
        for p in particle.tribe.particles.itervalues():
            yield p
        # This particle is always its own informer
        yield particle
        # Finally, there are outside connections as well
        for p in particle.ext_informers.itervalues():
            yield p

    def new_tribe( self ):
        t = _Tribe( self, self.curtribe )
        self.curtribe += 1
        self.tribes[t.idx] = t
        return t

    def kill_tribe( self, tribe ):
        del self.tribes[tribe.idx]

    def num_neighbors( self, particle ):
        return 1 + len(particle.tribe.particles) + len(particle.ext_informers)

    def num_connections( self ):
        nc = 0
        for p in self.particles.itervalues():
            nc += self.num_neighbors( p )
        return nc

    def _restructure( self ):
        """Perform tribal restructuring based on goodness and badness of tribes
        and particles.
        """
        # Note that Clerc does some strange things:
        #   * If a tribe is not bad, he'll randomly assign it to be bad anyway.
        #   * He checks bad tribes and generates particles before removing them.

        new_tribe = None

        # Note that we iterate over a copy of the tribes, since we may be
        # adding them.

        for t in self.tribes.values():
            if not t.is_good():
                # This one is bad, so we get to generate a new particle and
                # attach it to the best particle in the tribe
                best, worst = t.extreme_particles()

                # If there is no tribe for the particle, we create an empty one
                if new_tribe is None:
                    new_tribe = self.new_tribe()

                new_particle = self.make_particle( new_tribe )

                # Now add them as external informers to each other
                new_particle.ext_informers[best.idx] = best
                best.ext_informers[new_particle.idx] = new_particle

                # Finally emit this new particle.
                yield new_particle

        # We have created a bunch of particles.  Now we get to remove the worst
        # from each good tribe. Note that we don't emit anything here since we
        # only emit when a function evaluation occurs.
        deltribes = []
        for t in self.tribes.itervalues():
            if t.is_good():
                # Here we remove a particle, the worst of the tribe.  If there
                # is only one particle, then we only remove it if it has a
                # better informer.
                best, worst = t.extreme_particles()
                do_removal = True
                if best is worst:
                    # Only one particle, so remove only if it has a better
                    # informer.  Also, set that external informer to the new
                    # best, which would normally belong to the tribe.
                    best = self.bestneighbor( worst )
                    do_removal = self.is_better( best.bestval, worst.bestval )
                    deltribes.append(t)

                if do_removal:
                    # This removes the particle from all internal lists and
                    # from its tribe.
                    self.kill_particle( worst )
                    # Relink every external informer this particle to be
                    # informers of 'best' (and vice versa)
                    for neighbor in worst.ext_informers.itervalues():
                        assert neighbor is not worst

                        # Kill self from the neighbor's informer list
                        if worst.idx in neighbor.ext_informers:
                            del neighbor.ext_informers[worst.idx]
                        # Add a symmetric link between best and neighbor
                        if neighbor is not best:
                            neighbor.ext_informers[best.idx] = best
                            best.ext_informers[neighbor.idx] = neighbor

        for t in deltribes:
            self.kill_tribe( t )

#------------------------------------------------------------------------------
