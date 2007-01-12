#!/usr/bin/python -tt

from __future__ import division


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
