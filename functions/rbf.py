from __future__ import division
from itertools import izip, chain
import math
import _general


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

    def setup(self, vec=None):
        """Initialize the RBF function.

        Allows an option vec option to specify the generating vector
        (otherwise such a vector is randomly generated using the seed).
        """
        super(RBF, self).setup()
        self._set_constraints(((0,100),) * self.dims)
        self.nbases = int(self.dims / (1 + 2 * self.inputdims))

        import random

        if vec:
            self.generating_vec = vec
        else:
            rand = random.Random(self.seed)
            self.initialize_vec(rand)
            from sys import stderr
            print >>stderr, 'Vector used to generate points:', \
                    ','.join(str(x) for x in self.generating_vec)
        # Use a second Random instance so that the set of generated data
        # is identical regardless of where the generating vector came from.
        rand = random.Random(self.seed + 1)
        self.generate_data(rand)

    def __call__(self, vec):
        """Evaluate sum squared error."""
        sumsqerr = 0.0
        for point in self.datapoints:
            output = point[-1]
            sumsqerr += (output - self.net_value(vec, point)) ** 2
        return sumsqerr

    def net_value(self, vec, point):
        """Return the value of the RBF network at a given point."""
        # Sum over the individual basis functions.
        return sum(self.rbf_value(func, point) for func in \
                        itergroup(vec, 2 * self.inputdims + 1))

    def rbf_value(self, params, point):
        """Return the value of a single basis function at a given point."""
        output_weight = params[0]
        param_iter = itergroup(params[1:], 2)
        input_weight_multiplier = self.input_weight_multiplier()

        # Sum over the dimensions of the data point.
        total = 0
        for ((weight, center), x) in izip(param_iter, point):
            # Penalty for the [inverse] weight going to 0 or below.
            if weight < 0.01:
                total += (weight - 0.01) ** 2
                weight = 0.01
            # Penalty for leaving the feasible region.
            if x < 0:
                total += x ** 2
            elif x > 100:
                total += (x - 100) ** 2
            total += math.exp(-input_weight_multiplier / weight
                    * (x - center) ** 2)
        return output_weight * total

    def input_weight_multiplier(self):
        """Convert from range [0, 100] to a "nice-looking" range."""
        return self.nbases / self.inputdims / 20

    def initialize_vec(self, rand):
        """Create a generating vector."""
        vec = []
        for i in xrange(self.nbases):
            # Note: expected value of gamma is (alpha * beta)
            output_weight = rand.gammavariate(20, 2)
            vec.append(output_weight)
            for j in xrange(self.inputdims):
                input_weight = rand.gammavariate(20, 2)
                center = rand.uniform(0, 100)
                vec.append(input_weight)
                vec.append(center)
        self.generating_vec = tuple(vec)

    def generate_data(self, rand):
        """Generate some points."""
        from amlpso.cubes.cube import Cube
        inputs_constraints = ((0, 100),) * self.inputdims
        inputs_cube = Cube(inputs_constraints)
        vec = self.generating_vec

        datapoints = []
        for i in xrange(self.npoints):
            point = inputs_cube.random_vec(rand)
            point_and_value = tuple(point) + (self.net_value(vec, point),)
            datapoints.append(point_and_value)
        self.datapoints = datapoints


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


def vec_to_rbf(inputdims, vec):
    """Get an easy to use function for the given number of input dimensions and
    the given vec."""
    try:
        vec = tuple(float(s) for s in vec.split(','))
    except AttributeError:
        pass
    if len(vec) % (2 * inputdims + 1) != 0:
        raise ValueError('Wrong number of input dimensions in given vec!')
    rbf = RBF(dims=len(vec), inputdims=inputdims)
    rbf.setup(vec=vec)
    return rbf


def get_rbf_plot_func(rbf):
    """Get an easy to use function for the given RBF instance."""
    vec = rbf.generating_vec
    if rbf.inputdims == 1:
        def function(x):
            return rbf.net_value(vec, (x,))
    else:
        def function(x):
            return rbf.net_value(vec, x)
    return function


def generate_points(nbases, points, inputdims, randomseed):
    """Create a random RBF network and generate points with it."""

    dims = nbases * (1 + 2 * inputdims)
    rbf = RBF(dims=dims, inputdims=inputdims, npoints=points, seed=randomseed)
    rbf.setup()

    for point in rbf.datapoints:
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
