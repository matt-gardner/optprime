"""Outputter classes which determine the format of swarm status messages."""


from __future__ import division
import sys
from mrs.param import Param, ParamObj

VALID_ARGS = frozenset(('iteration', 'particles', 'best'))


class Output(ParamObj):
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
        print best.iters, best.pbestval
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

    args = frozenset(('best',))

    def start(self):
        from datetime import datetime
        self.last_iter = 0
        self.last_time = datetime.now()

    def __call__(self, **kwds):
        best = kwds['best']
        iteration = best.iters
        if iteration <= 0: return
        from datetime import datetime
        now = datetime.now()
        delta = now - self.last_time
        seconds = (delta.days * 86400 + delta.seconds
                + delta.microseconds / 1000000)

        time_per_iter = seconds / (iteration - self.last_iter)
        print iteration, best.pbestval, \
                ",".join([str(x) for x in best.pbestpos]), \
                "; Time:", time_per_iter
        sys.stdout.flush()

        self.last_time = now
        self.last_iter = iteration


class Swarm(Output):
    """Outputs the value, pos., best value, and best pos. for each particle."""

    args = frozenset(('particles',))

    def __call__(self, **kwds):
        particles = kwds['particles']
        print particles[0].iters, len(particles)
        for part in particles:
            print part.value, ','.join(str(x) for x in part.pos), \
                    part.pbestval, ','.join(str(x) for x in part.pbestpos)
        print


class SwarmRepr(Output):
    """Outputs the value, pos., best value, and best pos. for each particle."""

    args = frozenset(('particles',))

    def __call__(self, **kwds):
        particles = kwds['particles']
        print particles[0].iters, len(particles)
        for part in particles:
            print repr(part)
            print
        print


class Stats(Output):
    """Outputs stats about how many particles have been updated."""

    args = frozenset(('iteration', 'best', 'particles'))

    def start(self):
        self.num_recent = 4
        self.iters = 0
        self.not_updated = 0
        self.stagnant_iters = 0
        self.last_val = 0
        self.counter = dict()
        self.changes = 0
        self.bestid = 0
        self.consistent_changes = 0
        self.recently_seen_changes = 0
        self.recentlyseen = []
        self.num_pbest_updates = 0
        self.num_pbest_possible_updates = 0
        self.prevpbest = dict()
        self.prevnbest = dict()
        self.stagnant_count = dict()
        self.iters_stagnant = dict()
        self.num_particles = 0

    def __call__(self, **kwds):
        best = kwds['best']
        self.iters = kwds['iteration']
        particles = kwds['particles']
        if self.iters == 1:
            for particle in particles:
                self.prevpbest[particle.id] = 0
                self.stagnant_count[particle.id] = 0
                self.counter[particle.id] = 0
                self.iters_stagnant[particle.id] = 0
                self.prevnbest[particle.id] = 0
                self.num_particles += 1
        stagnant = True

        # Things to check for each particle at each iteration
        numstagnant = 0
        really_stagnant = 0
        for particle in particles:
            # Updated pbest
            if particle.pbestval != self.prevpbest[particle.id]:
                self.num_pbest_updates += 1
                self.iters_stagnant[particle.id] = 0
                stagnant = False
            else:
                # Updated nbest
                if particle.nbestval != self.prevnbest[particle.id]:
                    pass
                # Didn't update either - stagnant particle
                else: 
                    numstagnant += 1
                    self.stagnant_count[particle.id] += 1
                    self.iters_stagnant[particle.id] += 1
                    if self.iters_stagnant[particle.id] > 30:
                        really_stagnant += 1
            self.prevpbest[particle.id] = particle.pbestval
            self.prevnbest[particle.id] = particle.nbestval
            self.num_pbest_possible_updates += 1

        # Done checking each particle, check the global best and print output
        # gbest was not updated
        if best.pbestval == self.last_val:
            self.not_updated += 1
            if stagnant:
                self.stagnant_iters += 1
                print self.iters, best.pbestval, '\tStagnant!', '\tStagnant for more than '+\
                        '30 iterations:',really_stagnant
            else:
                print self.iters, best.pbestval, '\tStagnant particles:',numstagnant, \
                        '\tStagnant for more than 30 iterations:',really_stagnant
        # gbest was updated
        else:
            self.last_val = best.pbestval
            if best.id in self.recentlyseen:
                self.recently_seen_changes += 1
            if best.id == self.bestid:
                self.consistent_changes += 1
            else:
                self.recentlyseen.append(best.id)
                self.recentlyseen = self.recentlyseen[-self.num_recent:]
            self.bestid = best.id
            self.counter[best.id] += 1
            self.changes += 1
            print self.iters, best.pbestval, best.id, '\tStagnant particles:',\
                    numstagnant, '\tStagnant for more than 30 iterations:',\
                    really_stagnant

    def finish(self):
        print 'Stats:'
        print 'Individual Particles: (Percent stagnant, percent particle'+\
                ' was the gbest)'
        num_worthless = 0
        for id in self.stagnant_count:
            percentstag = self.stagnant_count[id]/self.iters
            if percentstag > .9 and self.counter[id] == 0:
                num_worthless += 1
            print id, percentstag, self.counter[id]/self.changes
        print 'Percent of iterations that gbest was not updated:'
        print self.not_updated/self.iters
        print 'Number of stagnant iterations:'
        print '%d (%f' % (self.stagnant_iters,
                100*self.stagnant_iters/self.iters)+'%)'
        print 'Pbest updates/possible pbest updates:'
        print self.num_pbest_updates/self.num_pbest_possible_updates
        print 'Percent of gbest updates that were by the same particle:'
        print self.consistent_changes/self.changes
        print 'Percent of gbest updates that were by recently seen particles:'
        print self.recently_seen_changes/self.changes
        print 'Percent of particles that updated pbest less than 10% of the time:'
        print num_worthless/len(self.counter)


class BranchStats(Output):
    """Outputs stats about which branch of execution particles took."""

    args = frozenset(('iteration', 'particles', 'best'))

    def __init__(self):
        self.iters = 0
        self.prevpbest = dict()
        self.prevnbest = dict()
        self.prevnbestid = dict()
        self.PnotN = dict()
        self.PandN = dict()
        self.PandNMe = dict()
        self.notPnotN = dict()
        self.notPbutN = dict()
        self.consistentnbest = dict()
        self.positions = dict()

    def __call__(self, **kwds):
        best = kwds['best']
        iteration = kwds['iteration']
        particles = kwds['particles']
        self.iters += 1
        if self.iters == 1:
            for particle in particles:
                self.prevpbest[particle.id] = 0
                self.prevnbest[particle.id] = 0
                self.prevnbestid[particle.id] = -1
                self.PnotN[particle.id] = 0
                self.PandN[particle.id] = 0
                self.PandNMe[particle.id] = 0
                self.notPnotN[particle.id] = 0
                self.notPbutN[particle.id] = 0
                self.consistentnbest[particle.id] = 0

        for particle in particles:
            self.positions[particle.pos] = particle.id

        for particle in particles:
            # Updated nbest
            if particle.nbestval != self.prevnbest[particle.id]:
                if self.prevnbestid[particle.id] == self.positions[particle.nbestpos]:
                    self.consistentnbest[particle.id] += 1
                if particle.nbestval == particle.pbestval:
                    self.PandNMe[particle.id] += 1
                else:
                    if particle.pbestval != self.prevpbest[particle.id]:
                        self.PandN[particle.id] += 1
                    else:
                        self.notPbutN[particle.id] += 1
            # Did not update nbest
            else:
                if particle.pbestval != self.prevpbest[particle.id]:
                    self.PnotN[particle.id] += 1
                else:
                    self.notPnotN[particle.id] += 1
            self.prevpbest[particle.id] = particle.pbestval
            self.prevnbest[particle.id] = particle.nbestval
            self.prevnbestid[particle.id] = self.positions[particle.nbestpos]
        print iteration, best.pbestval

    def finish(self):
        print '\nStats:'
        print '\nIndividual Particles: (notPnotN, PnotN, PandNMe, notPbutN, PandN)'
        avenotpnotn = []
        avepnotn = []
        avepandnme = []
        avenotpbutn = []
        avepandn = []
        for id in self.PnotN:
            notpnotn = self.notPnotN[id]/self.iters
            avenotpnotn.append(notpnotn)
            pnotn = self.PnotN[id]/self.iters
            avepnotn.append(pnotn)
            pandnme = self.PandNMe[id]/self.iters
            avepandnme.append(pandnme)
            notpbutn = self.notPbutN[id]/self.iters
            avenotpbutn.append(notpbutn)
            pandn = self.PandN[id]/self.iters
            avepandn.append(pandn)
            print id,'%.3f %.3f %.3f %.3f %.3f' % (notpnotn,pnotn,pandnme,notpbutn,pandn)

        print '\nAverages: (notPnotN, PnotN, PandNMe, notPbutN, PandN)'
        notpnotn = sum(avenotpnotn)/len(avenotpnotn)
        pnotn = sum(avepnotn)/len(avepnotn)
        pandnme = sum(avepandnme)/len(avepandnme)
        notpbutn = sum(avenotpbutn)/len(avenotpbutn)
        pandn = sum(avepandn)/len(avepandn)
        print '%.3f %.3f %.3f %.3f %.3f' % (notpnotn,pnotn,pandnme,notpbutn,pandn)

        print '\nIndividual Particles: (notP, P)'
        avenotp = []
        avep = []
        for id in self.PnotN:
            notp = (self.notPnotN[id]+self.notPbutN[id])/self.iters
            avenotp.append(notp)
            p = (self.PnotN[id]+self.PandNMe[id]+self.PandN[id])/self.iters
            avep.append(p)
            print id,'%.3f %.3f' % (notp,p)

        print '\nAverages: (notP, P)'
        notp = sum(avenotp)/len(avenotp)
        p = sum(avep)/len(avep)
        print '%.3f %.3f' % (notp,p)


# vim: et sw=4 sts=4
