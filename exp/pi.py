#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import bisect
import numpy
from six import b
import sys
import time

import mrs
from mrs import param

try:
    range = xrange
except NameError:
    pass


class PseudoParticle(object):
    def __init__(self, pos, value, iters):
        self.iters = iters
        self.value = value
        self.pos = pos
        self.pbestval = value
        self.pbestpos = pos


class PI(mrs.IterativeMR):
    """Experimental Prototype"""
    def __init__(self, opts, args):
        """Mrs Setup (run on both master and slave)"""

        super(PI, self).__init__(opts, args)

        self.function = param.instantiate(opts, 'func')
        self.topology = param.instantiate(opts, 'top')

        self.function.setup()
        self.topology.setup(self.function)

    ##########################################################################
    # Bypass Implementation

    def bypass(self):
        """Run a "native" version of PSO without MapReduce."""

        if not self.cli_startup():
            return 1

        # Perform simulation
        try:
            self.output = param.instantiate(self.opts, 'out')
            self.output.start()
            self.bypass_run()
            self.output.finish()
        except KeyboardInterrupt as e:
            print("# INTERRUPTED")
        return 0

    def bypass_run(self):
        """Performs PSO without MapReduce.

        Compare to the producer/consumer methods, which use MapReduce to do
        the same thing.
        """
        self.randomize_function_center()

        self.sample = []

        for iteration in range(1, 1 + self.opts.iters):
            value, pos = self.next()

            # Output phase.  (If freq is 5, output after iters 1, 6, 11, etc.)
            if self.output.freq and not ((iteration - 1) % self.output.freq):
                kwds = {}
                if 'iteration' in self.output.args:
                    kwds['iteration'] = iteration
                if 'particles' in self.output.args:
                    particle = PseudoParticle(pos, value, iteration)
                    kwds['particles'] = [particle]
                if 'best' in self.output.args:
                    value, pos = self.sample[0]
                    kwds['best'] = PseudoParticle(pos, value, iteration)
                self.output(**kwds)
                if self.stop_condition(self.sample[0][0]):
                    self.output.success()
                    return

    ##########################################################################
    # Helper Functions (shared by bypass and mrs implementations)

    def next(self):
        """Pick a new point and evaluate."""
        iteration = len(self.sample)
        if iteration < self.opts.num:
            rand = self.initialization_rand(iteration)
            pos = self.topology.cube.random_vec(rand)
        else:
            rand = self.pivot_rand(iteration)
            pivot_i = None
            while pivot_i is None or pivot_i >= iteration:
                pivot_i = rand.geometric(self.opts.pivot_p) - 1
            pivot_value, pivot_pos = self.sample[pivot_i]
            pivot_pos = numpy.array(pivot_pos, dtype=numpy.float64)

            # Note: y_ = (y.dot(u) / u.dot(u)) * u
            # is the orthogonal projection of y onto u.
            # And z = y - y_ is the component of y orthogonal to u.

            # Pick a random dimension.
            # In the future, this can instead be an arbitrary vector (or
            # even a path) onto which points will be projected.
            dim = rand.randint(self.function.dims)

            points = []
            for i, (value, pos) in enumerate(self.sample):
                if i == pivot_i:
                    continue
                y = value - pivot_value
                # Shift the pos relative to the pivot as the new origin.
                pos = numpy.array(pos, dtype=numpy.float64)
                shifted_pos = pos - pivot_pos
                # x is the projection of shifted_pos along the given dim.
                x = shifted_pos[dim]
                orth_comp = shifted_pos.copy()
                orth_comp[dim] = 0.0
                # Use the Manhattan distance
                dist = sum(abs(orth_comp))

                # For now, only use a limited number of closer points.
                bisect.insort(points, (dist, x, y))
                del points[100:]

            # Weighted Least Squares Regression.
            # In the future, use something more robust, like kriging.
            wX_list = []
            y_list = []
            max_dist = points[-1][0]
            for dist, x, y in points:
                # Set the weight (this may not be the best scheme, but at
                # least it avoids division by 0).
                if max_dist > 0:
                    w = 1 / (1 + 2 * dist / max_dist)
                else:
                    w = 1
                #print('w:', w, dist, max_dist)
                #x_list = [x, x**2, x**3]
                x_array = numpy.array([x, x**2])
                wX_list.append(w * x_array)
                y_list.append(y)
            wX_array = numpy.vstack(wX_list)
            #print('wX:', wX_array)
            #print('y:', y_list)
            (c1, c2), _, _, _ = numpy.linalg.lstsq(wX_array, y_list)
            #print('c1, c2:', c1, c2)

            new_x = -c1 / c2
            pos = pivot_pos.copy()
            pos[dim] += new_x

        # Evaluate the objective function
        value = self.function(pos)
        pos = list(pos)
        bisect.insort(self.sample, (value, pos))
        return value, pos

    def stop_condition(self, value):
        """Determines whether the stopping criteria has been met.

        In other words, whether any particle has succeeded (e.g., at 0).
        """
        if self.function.is_opt(value):
            return True
        else:
            return False

    def cli_startup(self):
        """Checks whether the repository is dirty and reports options.

        Returns True if startup succeeded, otherwise False.
        """
        from amlpso import cli

        # Check whether the repository is dirty.
        mrs_status = cli.GitStatus(mrs)
        amlpso_status = cli.GitStatus(sys.modules[__name__])
        if not self.opts.hey_im_testing:
            if amlpso_status.dirty:
                print(('Repository amlpso (%s) is dirty!'
                        '  Use --hey-im-testing if necessary.')
                        % amlpso_status.directory, file=sys.stderr)
                return False
            if mrs_status.dirty:
                print(('Repository mrs (%s) is dirty!'
                        '  Use --hey-im-testing if necessary.')
                        % mrs_status.directory, file=sys.stderr)
                return False

        # Report command-line options.
        if not self.opts.quiet:
            now = time.localtime()
            date = time.strftime("%a, %d %b %Y %H:%M:%S +0000", now)
            print('#', sys.argv[0])
            print('# Date:', date)
            print('# Git Status:')
            print('#   amlpso:', amlpso_status)
            print('#   mrs:', mrs_status)
            print("# Options:")
            for key, value in sorted(vars(self.opts).items()):
                print('#   %s = %s' % (key, value))
            self.function.master_log()
            sys.stdout.flush()

        return True

    ##########################################################################
    # Rand Setters

    # We define the rand offsets here, both for these rand setters, and for
    # those of all subclasses.  These really need to be unique, so let's put
    # them all in one place.
    PIVOT_OFFSET = 1
    INITIALIZATION_OFFSET = 2
    FUNCTION_OFFSET = 5

    def pivot_rand(self, i):
        """Returns a Random for the given particle id.

        This ensures that each run will have a unique initial swarm state.
        """
        return self.numpy_random(self.PIVOT_OFFSET, i)

    def initialization_rand(self, i):
        """Returns a Random for the given particle id.

        This ensures that each run will have a unique initial swarm state.
        """
        return self.random(self.INITIALIZATION_OFFSET, i)

    def randomize_function_center(self):
        """Sets a random function center."""
        rand = self.random(self.FUNCTION_OFFSET)
        self.function.randomize_center(rand)


##############################################################################
# Busywork

def update_parser(parser):
    """Adds PSO options to an OptionParser instance."""
    # Set the default Mrs implementation to Bypass (instead of MapReduce).
    parser.usage = parser.usage.replace('Serial', 'Bypass')
    parser.set_default('mrs', 'Bypass')

    parser.add_option('-q', '--quiet',
            dest='quiet', action='store_true',
            help='Refrain from printing version and option information',
            default=False,
            )
    parser.add_option('-v', '--verbose',
            dest='verbose', action='store_true',
            help="Print out verbose error messages",
            default=False,
            )
    parser.add_option('-i', '--iters',
            dest='iters', type='int',
            help='Number of iterations',
            default=100,
            )
    parser.add_option('-n', '--num',
            dest='num', type='int',
            help='Number of initial random values',
            default=10,
            )
    parser.add_option('--pivot-p',
            dest='pivot_p', type='float',
            help='Parameter of the geometric distribution for picking a pivot',
            default=0.4,
            )
    parser.add_option('-f', '--func', metavar='FUNCTION',
            dest='func', action='extend', search=['amlpso.functions'],
            help='Function to optimize',
            default='sphere.Sphere',
            )
    parser.add_option('-t', '--top', metavar='TOPOLOGY',
            dest='top', action='extend', search=['amlpso.topology'],
            help='Initialization parameters',
            default='Isolated',
            )
    parser.add_option('-o', '--out', metavar='OUTPUTTER',
            dest='out', action='extend', search=['amlpso.output'],
            help='Style of output',
            default='Basic',
            )
    parser.add_option('--hey-im-testing',
            dest='hey_im_testing', action='store_true',
            help='Ignore errors from uncommitted changes (for testing only!)',
            default=False,
            )

    return parser


if __name__ == '__main__':
    mrs.main(PI, update_parser)

# vim: et sw=4 sts=4
