"""Outputter classes which determine the format of swarm status messages."""


from __future__ import division
import sys
from mrs.param import Param

VALID_ARGS = frozenset(('iteration', 'particles', 'best'))


class Output(object):
    """Output the results of an iteration in some form.

    This class should be extended to be useful.  The `args` attribute defines
    which keyword arguments are required.  Some possible arguments, such as
    `best` and `swarm`, can be expensive to compute in some implementations.
    """
    args = frozenset()

    _params = dict(
            freq=Param(default=1, type='int',
                doc='Number of iterations per value output')
            )

    def __init__(self, *args, **kwds):
        super(Output, self).__init__(args, kwds)
        assert self.args.issubset(VALID_ARGS)

    def __call__(self, **kwds):
        raise NotImplementedError()

    def start(self):
        pass

    def finish(self):
        pass


class Basic(Output):
    """Outputs the best value."""

    args = frozenset(('best',))

    def __call__(self, **kwds):
        """Output the current state of the simulation.
        
        In this particular instance, it just dumps stuff out to stdout, and it
        only outputs the globally best value in the swarm.
        """
        best = kwds['best']
        print best.pbestval
        sys.stdout.flush()


class Pair(Output):
    """Outputs the iteration and best value."""

    args = frozenset(('best',))

    def __call__(self, **kwds):
        best = kwds['best']
        print iteration, best.pbestval
        sys.stdout.flush()


class Timer(Output):
    """Outputs the elapsed time for each iteration."""

    args = frozenset(('iteration',))

    def start(self):
        from datetime import datetime
        self.last_iter = 0
        self.last_time = datetime.now()

    def __call__(self, **kwds):
        iteration = kwds['iteration']
        if iteration <= 0: return
        # Find time difference
        from datetime import datetime
        now = datetime.now()
        delta = now - self.last_time
        seconds = (delta.days * 86400 + delta.seconds
                + delta.microseconds / 1000000)

        time_per_iter = seconds / (iteration - self.last_iter)
        print time_per_iter
        sys.stdout.flush()

        self.last_time = now
        self.last_iter = iteration


class Extended(Output):
    """Outputs the best value and best position."""

    args = frozenset(('best',))

    def __call__(self, **kwds):
        best = kwds['best']
        print best.pbestval, " ".join([str(x) for x in best.pbestpos])
        sys.stdout.flush()


class Everything(Output):
    """Outputs the iteration, best value, best position, and elapsed time."""

    args = frozenset(('best', 'iteration'))

    def start(self):
        from datetime import datetime
        self.last_iter = 0
        self.last_time = datetime.now()

    def __call__(self, **kwds):
        best = kwds['best']
        iteration = kwds['iteration']
        if iteration <= 0: return
        from datetime import datetime
        now = datetime.now()
        delta = now - self.last_time
        seconds = (delta.days * 86400 + delta.seconds
                + delta.microseconds / 1000000)

        time_per_iter = seconds / (iteration - self.last_iter)
        print iteration, best.pbestval, \
                ",".join([str(x) for x in best.pbestpos]), \
                "; Time: ", time_per_iter
        sys.stdout.flush()

        self.last_time = now
        self.last_iter = iteration


class Swarm(Output):
    """Outputs the value, pos., best value, and best pos. for each particle."""

    args = frozenset(('iteration', 'particles'))

    def __call__(self, **kwds):
        iteration = kwds['iteration']
        particles = kwds['particles']
        print iteration, len(particles)
        for part in particles:
            print part.val, ' '.join(str(x) for x in part.pos), \
                    part.pbestval, ' '.join(str(x) for x in part.pbestpos)
        print


class Stats(Output):
    """Outputs stats about how many particles have been updated."""

    args = frozenset(('iteration', 'best'))

    def __init__(self):
        self.iters = 0
        self.not_updated = 0
        self.last_val = 0
        self.counter = dict()
        self.changes = 0
        self.consistent_changes = 0
        self.recently_seen_changes = 0
        self.bestidx = 0
        self.recentlyseen = []
        self.num_recent = 4

    def __call__(self, **kwds):
        best = kwds['best']
        self.iters = kwds['iteration']
        if best.pbestval == self.last_val:
            self.not_updated += 1
            print self.iters, best.pbestval
        else:
            self.last_val = best.pbestval
            if best.idx in self.recentlyseen:
                self.recently_seen_changes += 1
            if best.idx == self.bestidx:
                self.consistent_changes += 1
            else:
                self.recentlyseen.append(best.idx)
                self.recentlyseen = self.recentlyseen[-self.num_recent:]
            self.bestidx = best.idx
            if best.idx not in self.counter:
                self.counter[best.idx] = 0
            self.counter[best.idx] += 1
            self.changes += 1
            print self.iters, best.pbestval, best.idx

    def finish(self):
        print 'Stats:'
        #for idx in self.counter:
            #print idx, self.counter[idx]/self.changes
        print 'Percent of iterations that best was not updated:'
        print self.not_updated/self.iters
        print 'Percent of best updates that were by the same particle:'
        print self.consistent_changes/self.changes
        print 'Percent of best updates that were by recently seen particles:'
        print self.recently_seen_changes/self.changes


# vim: et sw=4 sts=4
