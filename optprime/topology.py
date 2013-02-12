from __future__ import division
from mrs.param import ParamObj, Param

try:
    range = xrange
except NameError:
    pass

from .cube import Cube


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
        iscale = self.initscale
        ioffset = self.initoffset

        constraints = []
        vconstraints = []
        for cl, cr in func.constraints:
            natural_width = abs(cr - cl)
            center = (cl + cr) / 2 + natural_width * self.initoffset
            initialization_width = natural_width * self.initscale

            left = center - initialization_width / 2
            right = center + initialization_width / 2
            constraints.append((left, right))

            vconstraints.append((-initialization_width, initialization_width))

        self.cube = Cube(constraints)
        self.vcube = Cube(vconstraints)

    def newparticle(self, i, rand):
        """Returns a new particle with the given id.

        Particles are distributed uniformly within the constraints.
        """
        from .particle import Particle
        newpos = self.cube.random_vec(rand)
        newvel = self.vcube.random_vec(rand)
        p = Particle(id=i, pos=newpos, vel=newvel)
        return p

    def newparticles(self, rand):
        """Yields new particles.

        Particles are distributed uniformly within the constraints.  The
        generator stops after creating the configured number of particles.

        Note that the same rand instance is used for all of the particles.
        """
        for i in range(self.num):
            yield self.newparticle(i, rand)

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
        for i in range(1,self.neighbors+1):
            yield (particle.id + i) % self.num
            yield (particle.id - i) % self.num


class DRing(Ring):
    """Directed (one-way) Ring"""
    def iterneighbors(self, particle):
        if not self.noselflink:
            yield particle.id
        for i in range(1,self.neighbors+1):
            yield (particle.id + i) % self.num


class Complete(_Topology):
    """Complete (aka fully connected, gbest, or star)"""
    def iterneighbors(self, particle):
        # Yield all of the particles up to this one, and all after, then this
        # one last.
        for i in range(self.num):
            if not (i == particle.id and self.noselflink):
                yield i


class Rand(_Topology):
    """Random topology (pick n particles and send a message to them)"""
    _params = dict(
        neighbors=Param(default=2, type='int',
            doc='Number of neighbors to send to.'),
        )

    def iterneighbors(self, particle):
        randrange = particle.rand.randrange
        id = particle.id
        num = self.num
        if (self.neighbors == -1):
            neighbors = num
        else:
            neighbors = self.neighbors
        for i in range(neighbors):
            yield randrange(num)
        if not self.noselflink:
            yield id


