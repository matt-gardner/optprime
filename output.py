"""Outputter classes which determine the format of swarm status messages."""


from __future__ import division
import sys
from mrs.param import Param


class Output(object):
    """Output the results of an iteration in some form.

    This class should be extended to be useful.  Note that the require_all
    attribute reveals whether this outputter requires the full population or
    just the gbest.
    """
    require_all = False
    _params = dict(
            freq=Param(default=1, type='int',
                doc='Number of iterations per value output')
            )

    def __call__(self, soc, iters):
        raise NotImplementedError()

    def start(self):
        pass

    def finish(self):
        pass


class Basic(Output):
    """Outputs the best value."""
    def __call__(self, soc, iters):
        """Output the current state of the simulation.
        
        In this particular instance, it just dumps stuff out to stdout, and it
        only outputs the globally best value in the swarm.
        """
        best = soc.bestparticle()

        print best.bestval
        sys.stdout.flush()


class Pair(Output):
    """Outputs the iteration and best value."""
    def __call__(self, soc, iters):
        best = soc.bestparticle()
        print iters, best.bestval
        sys.stdout.flush()


class IterNumVal(Output):
    """Outputs the iteration, number of particles, and best value."""
    def __call__(self, soc, iters):
        best = soc.bestparticle()
        print iters, soc.numparticles(),  best.bestval
        sys.stdout.flush()


class Timer(Output):
    """Outputs the elapsed time for each iteration."""

    def start(self):
        from datetime import datetime
        self.last_iter = 0
        self.last_time = datetime.now()

    def __call__(self, soc, iters):
        if iters <= 0: return
        # Find time difference
        from datetime import datetime
        now = datetime.now()
        delta = now - self.last_time
        seconds = (delta.days * 86400 + delta.seconds
                + delta.microseconds / 1000000)

        time_per_iter = seconds / (iters - self.last_iter)
        print time_per_iter
        sys.stdout.flush()

        self.last_time = now
        self.last_iter = iters


class Extended(Output):
    """Outputs the best value and best position."""
    def __call__(self, soc, iters):
        best = soc.bestparticle()
        print best.bestval, " ".join([str(x) for x in best.bestpos])
        sys.stdout.flush()


class Everything(Output):
    """Outputs the iteration, best value, best position, and elapsed time."""
    def start(self):
        from datetime import datetime
        self.last_iter = 0
        self.last_time = datetime.now()

    def __call__(self, soc, iters):
        if iters <= 0: return
        from datetime import datetime
        now = datetime.now()
        delta = now - self.last_time
        seconds = (delta.days * 86400 + delta.seconds
                + delta.microseconds / 1000000)

        time_per_iter = seconds / (iters - self.last_iter)
        best = soc.bestparticle()
        print iters, best.bestval, " ".join([str(x) for x in best.bestpos]), \
                "; Time: ", time_per_iter
        sys.stdout.flush()

        self.last_time = now
        self.last_iter = iters


class Swarm(Output):
    """Outputs the value, pos., best value, and best pos. for each particle."""

    require_all = True

    def __call__(self, soc, iters):
        print iters, len(soc.particles)
        for part in soc.particles:
            print part.val, ' '.join(str(x) for x in part.pos), \
                    part.bestval, ' '.join(str(x) for x in part.bestpos)
        print


class Stats(Output):
    """Outputs stats about how many particles have been updated."""

    require_all = True

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

    def __call__(self, soc, iters):
        best = soc.bestparticle()
        self.iters += 1
        if best.bestval == self.last_val:
            self.not_updated += 1
            print iters, best.bestval
        else:
            self.last_val = best.bestval
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
            print iters, best.bestval, best.idx

    def finish(self):
        print 'Stats:'
        #for idx in self.counter:
            #print idx, self.counter[idx]/self.changes
        print 'Percent of iterations that gbest was not updated:'
        print self.not_updated/self.iters
        print 'Percent of gbest updates that were by the same particle:'
        print self.consistent_changes/self.changes
        print 'Percent of gbest updates that were by recently seen particles:'
        print self.recently_seen_changes/self.changes


# vim: et sw=4 sts=4
