#!/usr/bin/python -tt

from __future__ import division
import sys

# TODO: start using optparse's "type='choice'" and "choices=[]"


#------------------------------------------------------------------------------
# Parse the program arguments
#------------------------------------------------------------------------------
def gen_simple_options( parser, optprefix, helpname, dct ):
    """Generate options for a single set of varargs

    arguments:
    parser -- option parser object
    optprefix -- prefix to be placed on the option name (--optprefix-arg)
    helpname -- name of the argument thing in help (Function)
    dct -- the actual dictionary from the simulation module
    """

    for aname, arg in dct.iteritems():
        if isinstance(arg.default,bool):
            nostr = ''
            prefixhelp = ''
            if arg.default == True:
                prefixhelp = 'Turns off '
                nostr = 'no'
            parser.add_option('', '--%s-%s%s' % (optprefix,nostr,aname),
                dest='%s_%s' % (optprefix,aname),
                default=str(arg.default),
                action="store_const", const=str(not arg.default),
                help="%s%s optional setting: %s" % (
                    prefixhelp, helpname, str(arg),)
                )
        else:
            parser.add_option('', '--%s-%s' % (optprefix,aname),
                    dest='%s_%s' % (optprefix,aname),
                    default=str(arg.default),
                    help="%s optional setting: %s" % (
                        helpname,str(arg),)
                    )

#------------------------------------------------------------------------------
def gen_varargs_options( parser, optprefix, helpname, dct ):
    """Generate options for things that use varargs -- multiple mixed things

    arguments:
    parser -- option parser object
    optprefix -- prefix to be placed on the option name (--optprefix-arg)
    helpname -- name of the argument thing in help (Function)
    dct -- the actual dictionary from the simulation module
    """
    args = {}
    for tname, thing in dct.iteritems():
        for aname, arg in thing.args.iteritems():
            if aname not in args:
                args[aname] = [arg, [tname]]
            else:
                args[aname][1].append( tname )

    for aname, (arg, tnames) in args.iteritems():
        if len(tnames) == len(dct):
            availability = "all"
        else:
            availability = ", ".join(tnames)

        if isinstance(arg.default,bool):
            nostr = ''
            prefixhelp = ''
            if arg.default == True:
                prefixhelp = 'Turns off '
                nostr = 'no'
            parser.add_option('', '--%s-%s%s' % (optprefix,nostr,aname),
                dest='%s_%s' % (optprefix,aname),
                default=str(arg.default),
                action="store_const", const=str(not arg.default),
                help="%s%s optional setting: %s (Available for %s)" % (
                    prefixhelp,helpname,str(arg),availability,)
                )
        else:
            parser.add_option('', '--%s-%s' % (optprefix,aname),
                dest='%s_%s' % (optprefix,aname),
                default=str(arg.default),
                help="%s optional setting: %s (Available for %s)" % (
                    helpname,str(arg),availability,)
                )

#------------------------------------------------------------------------------
# BasicOutput class -- deals with output of the data
#------------------------------------------------------------------------------

# TODO: Exorcism needed.  I like Chris, but eval()ing input strings is evil!
def prefix_args(prefix, options):
    """Create a dictionary of args for the given prefix.

    Currently prefix is one of: func, motion, soc, sim.
    """
    args = {}
    for optname, opt in options.__dict__.iteritems():
        if optname.startswith( prefix + '_' ) and opt is not None:
            suffix = optname[len(prefix)+1:]
            try:
                args[suffix] = eval(getattr(options, optname))
            except (SyntaxError, NameError):
                args[suffix] = getattr(options, optname)

    return args

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

    def finish(self):
        pass

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

class StatsOutput(Output):
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


    def __call__( self, soc, iters ):
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



#------------------------------------------------------------------------------

outputtypes = dict((cls.__name__, cls) for cls in
        (BasicOutput, PairOutput, IterNumValOutput, TimerOutput,
            ExtendedOutput, SwarmOutput, OutputEverything, StatsOutput))


# vim: et sw=4 sts=4
