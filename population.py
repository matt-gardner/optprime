from mrs.param import Param

class Population(object):
    """Population of particles.
    
    One of the difficult things about implementing Population is that you
    can't count on always having access to the members of the population
    (e.g., Mrs PSO).
    """

    _params = dict(
        num=Param(default=2, type='int',
            doc='Number of particles in the swarm'),
        initspace=Param(default=1.0,
            doc='Size of initialization space (per dimension)'),
        initoffset=Param(default=0.0,
            doc='Offset of initialization space (per dimension)'),
        )

    def __init__(self, func):
        self.func = func

        ispace = self.initspace
        ioffset = self.initoffset

        constraints = [
            (cl+abs(cr-cl)*ioffset,cl + abs(cr-cl)*ioffset + (cr-cl)*ispace)
            for cl,cr in func.constraints
            ]
        sizes = [abs(cr-cl) * ispace for cl, cr in func.constraints]
        vconstraints = [(-s,s) for s in sizes]

        self.cube = Cube(constraints)
        self.vcube = Cube(vconstraints)

    def newparticles(self, numparts=None):
        """Iterator that successively returns new position/velocity pairs
        within a possibly given set of constraints.
        """
        rand = self.rand

        if numparts is None:
            numparts = self.nparts

        dims = self.func.dims

        c = self.cube
        vc = self.vcube

        for x in xrange(numparts):
            newpos = c.random_vec(rand)
            if self.randvels:
                newvel = vc.random_vec(rand)
            else:
                newvel = Vector([0] * dims)
            yield Particle(newpos, newvel)


# vim: et sw=4 sts=4
