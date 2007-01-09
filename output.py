#!/usr/bin/python -tt

#------------------------------------------------------------------------------
# BasicOutput class -- deals with output of the data
#------------------------------------------------------------------------------
class BasicOutput(object):
    def __call__( self, soc, iters ):
        """Output the current state of the simulation
        
        In this particular instance, it just dumps stuff out to stdout, and it
        only outputs the globally best value in the swarm.
        """
        best = soc.bestparticle()

        print best.bestval

class PairOutput(object):
    def __call__( self, soc, iters ):
        best = soc.bestparticle()
        print iters, best.bestval

class IterNumValOutput(object):
    def __call__( self, soc, iters ):
        best = soc.bestparticle()
        print iters, soc.numparticles(),  best.bestval

#------------------------------------------------------------------------------
class ExtendedOutput(object):
    def __call__( self, soc, iters ):
        best = soc.bestparticle()
        print best.bestval, " ".join([str(x) for x in best.bestpos])

outputtypes = dict((cls.__name__, cls) for cls in
        (BasicOutput, PairOutput, IterNumValOutput, ExtendedOutput))

# vim: et sw=4 sts=4
