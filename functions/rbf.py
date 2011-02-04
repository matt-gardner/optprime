from __future__ import division
from itertools import izip, chain
import math
import _general

#RBF_STDDEV = 10
RBF_STDDEV = 25


class RBF(_general._Base):
    """Radial Basis Function Network trainer

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

        if self.datafiles:
            datafile = self.datafiles.split()[0]
            self.datapoints = [tuple(float(field) for field in line.split(','))
                            for line in open(datafile)]
        else:
            self.datapoints = self.generate_data()

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

    def net_value(self, vec, point):
        """Return the value of the RBF network at a given point."""
        return sum(self.rbf_value(func, point) for func in \
                        itergroup(vec, 2 * self.data_dims + 1))

    def rbf_value(self, params, point):
        """Return the value of a single basis function at a given point."""
        output_weight = params[0]
        param_iter = itergroup(params[1:], 2)

        return output_weight * sum(
            (math.exp(-weight * (x - center) ** 2))
                for ((weight, center), x) in izip(param_iter, point))

    def generate_data(self):
        """Generate some points."""
        import random
        rand = random.Random(self.seed)

        self.data_dims = self.inputdims
        nbases = int(self.dims / (1 + 2 * self.data_dims))

        vec = []
        for i in xrange(nbases):
            output_weight = rand.gammavariate(5, 5)
            vec.append(output_weight)
            for j in xrange(self.data_dims):
                input_weight = rand.gammavariate(5, 5)
                center = rand.uniform(-50, 50)
                vec.append(input_weight)
                vec.append(center)
        from sys import stderr
        print >>stderr, 'Vector used to generate points:', \
                ','.join(str(x) for x in vec)

        datapoints = []
        inputs_constraints = ((-50, 50),) * self.data_dims
        inputs_cube = Cube(inputs_constraints)
        for i in xrange(self.npoints):
            point = inputs_cube.random_vec(rand)
            point_and_value = tuple(point) + (self.net_value(vec, point),)
            datapoints.append(point_and_value)
        return datapoints


def itergroup(iterator, count):
    """Iterate in groups of 'count' values. If there aren't enough values, the
    last result is padded with None."""
    iterator = iter(iterator)
    values_left = [1]
    def values():
        values_left[0] = 0
        for x in range(count):
            try:
                yield iterator.next()
                values_left[0] = 1
            except StopIteration:
                yield None
    while 1:
        value = tuple(values())
        if not values_left[0]:
            raise StopIteration
        yield value


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


def generate_points(nbases, points, inputdims, randomseed):
    """Create a random RBF network and generate points with it."""

    dims = nbases * (1 + 2 * inputdims)
    rbf = RBF(dims=dims)
    rbf.inputdims = inputdims
    rbf.npoints = points

    points = rbf.generate_data()
    for point in points:
        print ','.join(str(x) for x in point)


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
    parser.set_defaults(nbases=2, npoints=1000, inputdims=1, rand=42)

    options, args = parser.parse_args()
    nbases = options.nbases
    inputdims = options.inputdims
    npoints = options.npoints

    generate_points(nbases, npoints, inputdims, options.rand)

    #rbf = RBF(dims=3, datafile='rbftest.csv')
    #rbf.inputdims = 1
    #print rbf.net_value((1.0, 1.0, 0.0), 0.0)
    #print rbf((1.0, 1.0, 0.0))
