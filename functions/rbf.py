from __future__ import division
from itertools import izip
import _general
from iterextra import itergroup

class RBF(_general._Base):
    """Radial Basis Function Network

    The function takes an RBF and returns the sum squared error over the
    dataset in 'datafile'.

    The 'dims' parameter should be num_bases * (1 + 2 * num_input_dimensions)
    where num_input_dimensions is defined by the data found in 'datafile'.
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
            output = datapoint[-1]
            sumsqerr += (output - self.net_value(vec, point)) ** 2
        return sumsqerr

    def net_value( self, vec, point ):
        """Return the value of the RBF network at a given point."""
        return sum(self.rbf_value(func, point) for func in \
                        itergroup(vec, 2 * self.inputdims + 1))

    def rbf_value( self, params, point ):
        """Return the value of a single basis function at a given point."""
        output_weight = params[0]
        param_iter = itergroup(params[1:], 2)

        sq_dist = sum((weight * (x - center) ** 2) for (weight, center), x in \
                        izip(param_iter, point))

        return output_weight * stats.norm.pdf(sq_dist ** 0.5)
