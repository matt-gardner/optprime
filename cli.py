#!/usr/bin/python -tt

from __future__ import division
import sys


#------------------------------------------------------------------------------
# BasicOutput class -- deals with output of the data
#------------------------------------------------------------------------------
class Output(object):
    """Output the results of an iteration in some form.

    This class should be extended to be useful.  Note that the require_all
    attribute reveals whether this outputter requires the full population or
    just the gbest.
    """
    require_all = False

    def __call__(self, soc, iters):
        raise NotImplementedError()

class BasicOutput(Output):
    def __call__( self, soc, iters ):
        """Output the current state of the simulation
        
        In this particular instance, it just dumps stuff out to stdout, and it
        only outputs the globally best value in the swarm.
        """
        best = soc.bestparticle()

        print best.bestval
        sys.stdout.flush()

class PairOutput(Output):
    def __call__( self, soc, iters ):
        best = soc.bestparticle()
        print iters, best.bestval
        sys.stdout.flush()

class IterNumValOutput(Output):
    def __call__( self, soc, iters ):
        best = soc.bestparticle()
        print iters, soc.numparticles(),  best.bestval
        sys.stdout.flush()

class TimerOutput(Output):
    def __init__( self ):
        from datetime import datetime
        self.last_iter = 0
        self.last_time = datetime.now()

    def __call__( self, soc, iters ):
        if iters <= 0: return
        # Find time difference
        from datetime import datetime
        now = datetime.now()
        delta = now - self.last_time
        seconds = delta.days * 86400 + delta.seconds + delta.microseconds / 1000000

        time_per_iter = seconds / (iters - self.last_iter)
        print time_per_iter
        sys.stdout.flush()

        self.last_time = now
        self.last_iter = iters

class ExtendedOutput(Output):
    def __call__( self, soc, iters ):
        best = soc.bestparticle()
        print best.bestval, " ".join([str(x) for x in best.bestpos])
        sys.stdout.flush()

class OutputEverything(Output):
    def __init__(self):
        from datetime import datetime
        self.last_iter = 0
        self.last_time = datetime.now()

    def __call__( self, soc, iters ):
        if iters <= 0: return
        from datetime import datetime
        now = datetime.now()
        delta = now - self.last_time
        seconds = delta.days * 86400 + delta.seconds + delta.microseconds / 1000000

        time_per_iter = seconds / (iters - self.last_iter)
        best = soc.bestparticle()
        print iters, best.bestval, " ".join([str(x) for x in best.bestpos]), "; Time: ", time_per_iter
        sys.stdout.flush()

        self.last_time = now
        self.last_iter = iters


class SwarmOutput(Output):
    require_all = True

    def __call__( self, soc, iters ):
        print iters, len(soc.particles)
        for part in soc.particles:
            print part.val, ' '.join(str(x) for x in part.pos), \
                    part.bestval, ' '.join(str(x) for x in part.bestpos)
        print


#------------------------------------------------------------------------------

outputtypes = dict((cls.__name__, cls) for cls in
        (BasicOutput, PairOutput, IterNumValOutput, TimerOutput,
            ExtendedOutput, SwarmOutput, OutputEverything))


# vim: et sw=4 sts=4
