"""Outputter classes which determine the format of swarm status messages."""


from __future__ import division
import sys
from mrs.param import Param


class Output(object):
    """Output the results of an iteration in some form.

    This class should be extended to be useful.  Note that the require_all
    attribute reveals whether this outputter requires the full population or
    just the nbest.
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

        print best.pbestval
        sys.stdout.flush()


class Pair(Output):
    """Outputs the iteration and best value."""
    def __call__(self, soc, iters):
        best = soc.bestparticle()
        print iters, best.pbestval
        sys.stdout.flush()


class IterNumVal(Output):
    """Outputs the iteration, number of particles, and best value."""
    def __call__(self, soc, iters):
        best = soc.bestparticle()
        print iters, soc.numparticles(),  best.pbestval
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
        print best.pbestval, " ".join([str(x) for x in best.pbestpos])
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
        print iters, best.pbestval, " ".join([str(x) for x in best.pbestpos]), \
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
                    part.pbestval, ' '.join(str(x) for x in part.pbestpos)
        print


class Stats(Output):
    """Outputs stats about how many particles have been updated."""

    require_all = True

    def start(self):
        self.num_recent = 4
        self.iters = 0
        self.not_updated = 0
        self.stagnant_iters = 0
        self.last_val = 0
        self.counter = dict()
        self.changes = 0
        self.consistent_changes = 0
        self.recently_seen_changes = 0
        self.bestidx = 0
        self.recentlyseen = []
        self.num_pbest_updates = 0
        self.num_pbest_possible_updates = 0
        self.prevpbest = dict()

    def __call__(self, soc, iters):
        best = soc.bestparticle()
        self.iters += 1
        if self.iters == 1:
            for particle in soc.particles:
                self.prevpbest[particle.idx] = 0
        stagnant = True
        for particle in soc.particles:
            if particle.pbestval != self.prevpbest[particle.idx]:
                self.num_pbest_updates += 1
                self.prevpbest[particle.idx] = particle.pbestval
                stagnant = False
            self.num_pbest_possible_updates += 1
        if best.pbestval == self.last_val:
            self.not_updated += 1
            if stagnant:
                self.stagnant_iters += 1
                print iters, best.pbestval, 'Stagnant!'
            print iters, best.pbestval
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
            print iters, best.pbestval, best.idx

    def finish(self):
        print 'Stats:'
        print 'Percent of iterations that nbest was not updated:'
        print self.not_updated/self.iters
        print 'Number of stagnant iterations: %d (%d\%)' % (self.stagnant_iters,
                self.stagnant_iters/self.iters)
        print 'Percent of nbest updates that were by the same particle:'
        print self.consistent_changes/self.changes
        print 'Percent of nbest updates that were by recently seen particles:'
        print self.recently_seen_changes/self.changes


# vim: et sw=4 sts=4
