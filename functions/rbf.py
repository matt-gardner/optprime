from __future__ import division
from itertools import izip, chain
import _general
from iterextra import itergroup

try:
    from scipy import stats
    gaussian = stats.norm.pdf
except ImportError:
    from math import sqrt, exp, pi
    def gaussian(x,scale=1.0):
        return 1.0/(scale*sqrt(2*pi))*exp(-(x/scale)**2/2.0)

#RBF_STDDEV = 10
RBF_STDDEV = 25

class RBF(_general._Base):
    """Radial Basis Function Network

    The function takes an RBF and returns the sum squared error over the
    dataset in the first file in 'datafiles'.

    The 'dims' parameter should be num_bases * (1 + 2 * num_input_dimensions)
    where num_input_dimensions is defined by the data found in the datafile.
    The portion of 'vec' for each basis ends up looking like:
    (output_weight, weight1, center1, weight2, center2, ...)

    Note: we do not yet set the deviation/variance for each basis function.
    """
    from mrs.param import Param

    _params = dict(
            datafiles=Param(default='', 
                doc='File with training data (CSV format)'),
            npoints=Param(default=1000, type='int', 
                doc='Number of generated data points'),
            inputdims=Param(default=1, type='int', 
                doc='Dimensionality of generated points'),
            seed=Param(default=42, type='int', 
                doc='Random seed used to generate points')
            )

    def setup(self):
        super(RBF,self).setup()
        self._set_constraints(((-50,50),) * self.dims)
        self.data_dims = None
        try:
            datafile = self.datafiles.split()[0]
            self.datapoints = [tuple(float(field) for field in line.split(','))
                            for line in open(datafile)]
        except IndexError:
            self.datapoints = []

        # Debugging:
        #import os, tempfile
        #from subprocess import Popen, PIPE
        #self.debug_proc = Popen(('%s/bin/hadoop' % os.environ['HADOOP_HOME'],
        #    'dfs', '-put', '-', tempfile.mktemp()), stdin=PIPE)
        #example: print >>self.debug_proc.stdin, 'Hello!'

    def __call__(self, vec):
        """Evaluate sum squared error."""
        sumsqerr = 0.0
        for point in self.datapoints:
            # Figure out the dimensionality of our dataset by looking at the
            # first point in the file.
            if self.data_dims is None:
                self.data_dims = len(point) - 1
            output = point[-1]
            sumsqerr += (output - self.net_value(vec, point)) ** 2
        return sumsqerr

    def net_value( self, vec, point ):
        """Return the value of the RBF network at a given point."""
        return sum(self.rbf_value(func, point) for func in \
                        itergroup(vec, 2 * self.data_dims + 1))

    def rbf_value( self, params, point ):
        """Return the value of a single basis function at a given point."""
        output_weight = abs(params[0])
        param_iter = itergroup(params[1:], 2)

        sq_dist = sum((abs(weight) * (x - center) ** 2) for
                (weight, center), x in izip(param_iter, point))

        return output_weight * gaussian(sq_dist ** 0.5, scale=RBF_STDDEV)

    def tmpfiles(self):
        """Generate some points in a temporary file and return the filename."""
        import random
        rand = random.Random(self.seed)

        self.data_dims = self.inputdims
        nbases = int(self.dims / (1 + 2 * self.data_dims))

        # We'll limit constraints a bit for our generated data.
        center_constraints = []
        for i in xrange(nbases):
            center_constraints.append((0, 50))
            for j in xrange(self.data_dims):
                center_constraints.append((0, 50))
                center_constraints.append((-50, 50))
                # Use this if you want to make the RBF centers more evenly spaced:
                #center_constraints.append((-50 + 100 * i / nbases,
                #    -50 + 100 * (i + 1) / nbases))
        from amlpso.cubes.cube import Cube
        vec = Cube(center_constraints).random_vec(rand)
        from sys import stderr
        print >>stderr, 'Vector used to generate points:', \
                ','.join(str(x) for x in vec)

        import os, tempfile
        csvfd, csvfilename = tempfile.mkstemp()
        csvfile = os.fdopen(csvfd, 'w')
        inputs_constraints = ((-50, 50),) * self.data_dims
        inputs_cube = Cube(inputs_constraints)
        for i in xrange(self.npoints):
            point = inputs_cube.random_vec(rand)
            point_and_value = tuple(chain(point, (self.net_value(vec, point),)))
            self.datapoints.append(point_and_value)
            print >>csvfile, ','.join(str(x) for x in point_and_value)
        csvfile.close()
        return (csvfilename,)

def get_rbf_plot_func(inputdims, vec):
    """Get an easy to use function for the given number of input dimensions and
    the given vec."""
    try:
        split = vec.split(' ')
        if len(split) < 2:
            raise ValueError
        vec = tuple(float(s) for s in split)
    except (AttributeError, ValueError):
        split = vec.split(',')
        if len(split) < 2:
            raise ValueError
        vec = tuple(float(s) for s in split)
    if len(vec) % (2 * inputdims + 1) != 0:
        raise ValueError('Wrong number of input dimensions in given vec!')
    rbf = RBF()
    rbf.data_dims = inputdims
    if inputdims == 1:
        def function(x):
            return rbf.net_value(vec, (x,))
    else:
        def function(x):
            return rbf.net_value(vec, x)
    return function

def generate_points(bases, points, inputdims, randomseed):
    """Create a random RBF and generate points with it."""

    dims = nbases * (1 + 2 * inputdims)
    rbf = RBF(dims=dims)
    rbf.inputdims = inputdims
    rbf.npoints = points

    csvfilename = rbf.tmpfiles()[0]
    csvfile = open(csvfilename)
    for line in csvfile:
        print line,
    from os import unlink
    unlink(csvfilename)


if __name__ == '__main__':
    # Generate data, picking a random RBF
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-b', '--bases', dest='nbases', type='int',
            help='Number of Basis Functions')
    parser.add_option('-n', '--npoints', dest='npoints', type='int',
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

    generate_points(nbases, npoints, inputdims, options.rand)

    #rbf = RBF(dims=3, datafile='rbftest.csv')
    #rbf.inputdims = 1
    #print rbf.net_value((1.0, 1.0, 0.0), 0.0)
    #print rbf((1.0, 1.0, 0.0))
