class Population(object):
    """Population of particles.
    
    One of the difficult things about implementing Population is that you
    can't count on always having access to the members of the population
    (e.g., Mrs PSO).
    """

    _params = dict(
        initspace=Param(default=1.0,
            doc='Size of initialization space (per dimension)'),
        initoffset=Param(default=0.0,
            doc='Offset of initialization space (per dimension)'),
        kmeaniters=Param(default=0, doc='K-Means initialization iterations'),
        )

    def __init__(self, func):
        # Save these for later.
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

        corner1, corner2 = zip(*constraints)
        self.diaglen = abs(Vector(corner1) - Vector(corner2))

    def newparticle(self, **args):
        args['numparts'] = 1
        piter = self.iternewparticles(**args)
        return piter.next()

    def iternewparticles(self, **args):
        func = self.func
        for pos, vel in self.iternewstates(**args):
            val = func(pos)
            yield Particle(pos, vel, val)



# vim: et sw=4 sts=4
