from __future__ import division
from mrs.param import ParamObj, Param

# TODO: should initscale and initoffset be moved into Function??
class _Topology(ParamObj):
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

    def newparticles(self, batch, rand):
        """Yields new particles.

        Particles are distributed uniformly within the constraints.  The
        generator stops after creating the configured number of particles.
        """
        from particle import Particle
        for i in xrange(self.num):
            newpos = self.cube.random_vec(rand)
            newvel = self.vcube.random_vec(rand)
            p = Particle(pid=i, pos=newpos, vel=newvel)
            p.batches = batch
            yield p

    def iterneighbors(self, particle):
        """Yields the particle ids of the neighbors of the give particle."""
        raise NotImplementedError


class Isolated(_Topology):
    """Independent isolated particles."""

    def iterneighbors(self, particle):
        if not self.noselflink:
            yield particle.id


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


