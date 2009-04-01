#!/usr/bin/env python

from mrs import param
from param import ParamObj, Param

class _SpecMethod(ParamObj):
    """A method for performing speculative execution.
    
    Things such as how to pick the speculative branch to accept, how much
    pruning to do, and how to determine which neighbors to accept are defined
    here.
    """
    
    def pick_child(self, particle, neighbors, children):
        """Given a particle, its neighbors, and its children, decide which 
        child to return."""
        raise NotImplementedError

    def pick_neighbor_child(self, neighbor, neighbor_messages):
        """Given a neighbor and messages for the neighbor (including its
        children), decide which child of the neighbor to return."""
        raise NotImplementedError
        

    def itermessages(self, particle):
        """Given a particle, return the set of particles to whom it must
        send messages."""
        raise NotImplementedError

    def generate_children(self, particle, neighbors):
        """Yield the speculative children of a particle given it and its
        neighbors.  Often uses an external pruning class."""
        raise NotImplementedError

class ReproducePSO(_SpecMethod):
    """Perform speculative execution in such a way that the original PSO is
    reproduced exactly, just two iterations at a time.  A lot of communication
    is required to make sure the behavior is exactly the same as the original
    PSO, and any good information from speculative particles is ignored except
    on the branch that standard PSO would have taken.
    """

    def pick_child(self, particle, neighbors, children):
        # Look at the messages to see which branch you actually took
        best = self.findbest(messages, comparator)
        if best:
            particle.nbest_cand(best.position, best.value, comparator)
        print 'Particle is now:',repr(particle)

        # If you updated your pbest, then your current value will be your
        # pbestval
        updatedpbest = particle.pbestval == particle.value
        if particle.nbestval == best.value:
            bestid = best.sender
        else:
            bestid = -1

        # Look through the children and pick the child that corresponds to the
        # branch you took
        for child in children:
            if child.specpbest == updatedpbest and \
                child.specnbestid == bestid:
                    return child.make_real_particle()
        raise RunTimeError("Didn't find a child that matched the right branch!")


class _Pruning(ParamObj):
    """Speculative Pruning
    
    You have an infinite tree of possible executions.  This class defines
    one method, generate_children, that decides how far to go down the 
    tree, and in what way.
    """

    def generate_children(self, particle, neighbors):
        """Given a particle and its neighbors, generate the particle's
        speculative children."""
        raise NotImplementedError

class OneCompleteIteration(_Pruning):

    def generate_children(self, particle, neighbors):
        """Given the particle and neighbors at iteration i, yield all possible
        speculative children for iteration i+1."""
        # Create children guessing that pbest was not updated
        child = SEParticle(p, specpbest=False, specnbestid=-1)
        self.just_move(child)
        yield child
        for n in neighbors:
            if n.id != p.id:
                child = SEParticle(p, specpbest=False, specnbestid=n.id)
                child.nbestpos = n.pos
                self.just_move(child)
                yield child

        # Create children guessing that pbest was updated
        child = SEParticle(p, specpbest=True, specnbestid=-1)
        child.pbestpos = p.pos
        self.just_move(child)
        yield child
        for n in neighbors:
            child = SEParticle(p, specpbest=True, specnbestid=n.id)
            child.nbestpos = n.pos
            child.pbestpos = p.pos
            self.just_move(child)
            yield child

# vim: et sw=4 sts=4
