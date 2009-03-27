from __future__ import division
from mrs.param import Param

# TODO: should initscale and initoffset be moved into Function??
class _Topology(object):
    """Swarm Topology

    A Topology object can create particles and determine neighborhoods, but it
    does not hold swarm state.  Thus, multiple independent swarms should be
    able to share the same Topology object.

    Although we usually think of the members of a topology as particles, we
    occasionally build up topologies of swarms, too.  In most cases, the
    distinction is irrelevant to the topology.
    """
    _params = dict(
        num=Param(default=20, type='int', shortopt='-n',
            doc='Number of particles in the swarm'),
        noselflink=Param(type='bool',
            doc='Do not include self in neighborhood'),
        initscale=Param(default=1.0,
            doc='Scale factor of initialization space (per dimension)'),
        initoffset=Param(default=0.0,
            doc='Offset of initialization space (per dimension)'),
        )

    def setup(self, func):
        from cubes.cube import Cube

        ispace = self.initscale
        ioffset = self.initoffset

        # TODO: Make sure this is correct.  Shouldn't both the left and the
        # right constraints need an ispace term?  The inner formula should
        # even be split off into a separate testable method.
        constraints = [
            (cl+abs(cr-cl)*ioffset,cl + abs(cr-cl)*ioffset + (cr-cl)*ispace)
            for cl,cr in func.constraints
            ]
        sizes = [abs(cr-cl) * ispace for cl, cr in func.constraints]
        vconstraints = [(-s,s) for s in sizes]

        self.cube = Cube(constraints)
        self.vcube = Cube(vconstraints)

    def newparticles(self, batch, rand, initid=0):
        """Yields new particles.

        Particles are distributed uniformly within the constraints.  The
        generator stops after creating the configured number of particles.
        """
        from particle import Particle
        for i in xrange(self.num):
            newpos = self.cube.random_vec(rand)
            newvel = self.vcube.random_vec(rand)
            p = Particle(pid=(i + initid), pos=newpos, vel=newvel)
            p.batches = batch
            yield p

    def iterneighbors(self, particle):
        """Yields the particle ids of the neighbors of the give particle."""
        raise NotImplementedError


class Ring(_Topology):
    """Bidirectional Ring (aka lbest)"""
    _params = dict(
        neighbors=Param(default=1, type='int',
            doc='Number of neighbors to send to on each side'),
        )

    def iterneighbors(self, particle):
        if not self.noselflink:
            yield particle.id
        for i in xrange(1,self.neighbors+1):
            yield (particle.id + i) % self.num
            yield (particle.id - i) % self.num


class DRing(Ring):
    """Directed (one-way) Ring"""
    def iterneighbors(self, particle):
        if not self.noselflink:
            yield particle.id
        for i in xrange(1,self.neighbors+1):
            yield (particle.id + i) % self.num


class Complete(_Topology):
    """Complete (aka fully connected, gbest, or star)"""
    def iterneighbors(self, particle):
        # Yield all of the particles up to this one, and all after, then this
        # one last.
        for i in xrange(self.num):
            if not (i == particle.id and self.noselflink):
                yield i


class Rand(_Topology):
    """Random topology (pick n particles and send a message to them)"""
    _params = dict(
        neighbors=Param(default=-1, type='int',
            doc='Number of neighbors to send to.  Default of -1 means all '+
            'particles'),
        )

    def iterneighbors(self, particle):
        randrange = particle.rand.randrange
        pid = particle.id
        num = self.num
        if (self.neighbors == -1):
            neighbors = num
        else:
            neighbors = self.neighbors
        for i in xrange(neighbors):
            yield randrange(num)
        if not self.noselflink:
            yield pid


class Islands(_Topology):
    _params = dict(
        num_islands=Param(default=2, type='int',
            doc='Number of islands to use'),
        )

    def iterneighbors(self, particle):
        # Particles are grouped into n islands, and communicate with all members
        # on the island, and no one else
        pid = particle.id
        num_particles = self.num
        islands = self.num_islands
        if num_particles % islands != 0:
            raise ValueError('Uneven split between islands! '+
            'num_particles % num_islands should be zero')
        step_size = int(num_particles/islands)
        for i in xrange(islands):
            if pid in xrange(i*step_size, i*step_size + step_size):
                for j in xrange(i*step_size, i*step_size + step_size):
                    yield j


class CommunicatingIslands(_Topology):
    _params = dict(
        num_islands=Param(default=4, type='int',
            doc='Number of islands to use'),
        iterations_per_communication=Param(default=50, type='int',
            doc='Number of iterations inbetween each inter-island communication'),
        type_of_communication=Param(default='Ring',
            doc='The sociometry to use at each inter-island communication (only'
            +' Ring and Random'),
        neighbors=Param(default=-1, type='float',
            doc='Number of nieghbors to communicate with during inter-island '+\
                    'communication. -1 means everyone.'),
        )

    def iterneighbors(self, particle):
        # Particles are grouped into n islands, and communicate with all members
        # on the island, and no one else
        pid = particle.id
        iter = particle.iters
        num = self.num
        islands = self.num_islands
        if num % islands != 0:
            raise ValueError('Uneven split between islands! '+
            'num % num_islands should be zero')
        step_size = int(num/islands)
        if iter%self.iterations_per_communication == 0:
            # It's time to tell the other islands what's going on
            # There are probably smarter ways to do this communication, but
            # this is at least a temporary solution for the serial code
            if self.neighbors == -1:
                for i in range(self.num):
                    yield i
            elif self.type_of_communication == 'Ring':
                 for i in range(1, self.neighbors+1):
                     yield (pid+i) % num
                     yield (pid-i) % num
                 if not self.noselflink:
                     yield pid
            elif self.type_of_communication == 'Random':
                randrange = particle.rand.randrange()
                for i in xrange(self.neighbors):
                    yield randrange(num)
                if not self.noselflink:
                    yield pid
        else:
            for i in xrange(islands):
                if pid in xrange(i*step_size, i*step_size + step_size):
                    for j in xrange(i*step_size, i*step_size + step_size):
                        yield j





