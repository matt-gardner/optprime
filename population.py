class Population(object):
    """Population of particles.
    
    One of the difficult things about implementing Population is that you
    can't count on always having access to the members of the population
    (e.g., Mrs PSO).
    """

    def __init__(self, constraints):
        self.constraints = constraints

    def iternewstates(self, **args):
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


# vim: et sw=4 sts=4
