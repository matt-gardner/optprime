from __future__ import division
from itertools import izip, chain
from scipy import stats
import _general
from iterextra import itergroup

RBF_STDDEV = 10

class RBF(_general._Base):
    """Radial Basis Function Network

    The function takes an RBF and returns the sum squared error over the
    dataset in 'datafile'.

    The 'dims' parameter should be num_bases * (1 + 2 * num_input_dimensions)
    where num_input_dimensions is defined by the data found in 'datafile'.
    The portion of 'vec' for each basis ends up looking like:
    (output_weight, weight1, center1, weight2, center2, ...)

    Note: we do not yet set the deviation/variance for each basis function.
    """
    _args = [('datafile', '/var/hadoop/rbfdata.csv', \
                    'File with training data (CSV format)')]
    def __init__( self, *args, **kargs):
        super(RBF,self).__init__( *args, **kargs )
        self._set_constraints( ((-50,50),) * self.dims )
        self.inputdims = None

    def __call__( self, vec ):
        """Evaluate sum squared error."""
        datapoints = (tuple(float(field) for field in line.split(','))
                        for line in open(self.datafile))
        sumsqerr = 0.0
        for point in datapoints:
            # Figure out the dimensionality of our dataset by looking at the
            # first point in the file.
            if self.inputdims is None:
                self.inputdims = len(point) - 1
            output = point[-1]
            sumsqerr += (output - self.net_value(vec, point)) ** 2
        return sumsqerr

    def net_value( self, vec, point ):
        """Return the value of the RBF network at a given point."""
        return sum(self.rbf_value(func, point) for func in \
                        itergroup(vec, 2 * self.inputdims + 1))

    def rbf_value( self, params, point ):
        """Return the value of a single basis function at a given point."""
        output_weight = abs(params[0])
        param_iter = itergroup(params[1:], 2)

        sq_dist = sum((abs(weight) * (x - center) ** 2) for
                (weight, center), x in izip(param_iter, point))

        return output_weight * stats.norm.pdf(sq_dist ** 0.5, scale=RBF_STDDEV)


if __name__ == '__main__':
    # Generate data, picking a random RBF
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-b', '--bases', dest='nbases', type='int',
            help='Number of Basis Functions')
    parser.add_option('-p', '--points', dest='npoints', type='int',
            help='Number of Points')
    parser.add_option('-d', '--inputdims', dest='inputdims', type='int',
            help='Number of Input Dimensions')
    parser.add_option('-r', '--randomseed', dest='rand', type='int',
            help='Random Seed for Bases')
    parser.set_defaults(nbases=2, npoints=1000, inputdims=1, rand=0)

    options, args = parser.parse_args()
    nbases = options.nbases
    inputdims = options.inputdims
    npoints = options.npoints
    import random
    rand = random.Random(options.rand)

    dims = nbases * (1 + 2 * inputdims)
    rbf = RBF(dims=dims)
    rbf.inputdims = inputdims

    # We'll limit constraints a bit for our generated data.
    center_constraints = []
    for i in xrange(nbases):
        center_constraints.append((0, 50))
        for j in xrange(inputdims):
            center_constraints.append((0, 50))
            center_constraints.append((-50, 50))
            # Use this if you want to make the RBF centers more evenly spaced:
            #center_constraints.append((-50 + 100 * i / nbases,
            #    -50 + 100 * (i + 1) / nbases))
    from aml.opt.cubes.cube import Cube
    vec = Cube(center_constraints).random_vec(rand)
    from sys import stderr
    print >>stderr, 'Vec:', ','.join(str(x) for x in vec)

    inputs_constraints = ((-50, 50),) * inputdims
    inputs_cube = Cube(inputs_constraints)
    for i in xrange(npoints):
        point = inputs_cube.random_vec(rand)
        print ','.join(str(x) for x in chain(point, (rbf.net_value(vec, point),)))

    #rbf = RBF(dims=3, datafile='rbftest.csv')
    #rbf.inputdims = 1
    #print rbf.net_value((1.0, 1.0, 0.0), 0.0)
    #print rbf((1.0, 1.0, 0.0))
