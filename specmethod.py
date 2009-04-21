#!/usr/bin/env python

from mrs.param import ParamObj, Param
from particle import Particle, SEParticle, Dummy

class _SpecMethod(ParamObj):
    """A method for performing speculative execution.
    
    Things such as how to pick the speculative branch to accept, how much
    pruning to do, and how to determine which neighbors to accept are defined
    here.
    """

    def setup(self, specex, pruner):
        self.specex = specex
        self.pruner = pruner

    ##########################################################################
    # Methods that must be overridden
    
    def pick_child(self, particle, it1messages, children):
        """Returns particle at iteration 2 (minus nbest).

        particle is at iteration 1, children are all speculative children of
        particle (at iteration 2).  it1messages contain all messages that 
        particle received in the Reduce phase.
        """
        raise NotImplementedError

    def pick_neighbor_children(self, particle, it1messages, it2messages):
        """Returns particle's neighbors at iteration 2 (minus nbest).

        it1messages contains all iteration 1 messages particle received, not 
        all of which are actual neighbors of particle.  it2messages contains
        all iteration 2 (speculative) messages particle received.  Pick the
        actual neighbors from it1messages, and pick their children from 
        it2messages.  Update the children's pbest and return them.
        """
        raise NotImplementedError

    def update_neighbor_nbest(self, neighbors, it1messages, it2messages):
        """Make finished it2 particles out of neighbors by updating their nbest.

        Neighbors is the set of neighbors at iteration 2 minus their nbest.
        Get whatever info is needed from it1messages and it2messages to update
        the nbest of all particles in neighbors.
        
        Some speculative methods skip this step to save on messages that
        need to be passed, so the function would be empty.
        """
        raise NotImplementedError

    def message_ids(self, particle):
        """Return the set of particles to whom a particle must send messages."""
        raise NotImplementedError

    ##########################################################################
    # Methods that probably should not be overridden.  

    def generate_children(self, particle, neighbors):
        """Yield the speculative children of a particle given it and its
        neighbors.  Often uses an external pruning class."""
        return self.pruner.generate_children(particle, neighbors)

    def get_neighbors(self, particle, messages):
        """Get the neighbors of particle from the set of messages.

        It is a bit less work if the topology is symmetric, but this works 
        in general.  
        
        Also, messing with the neighbors' iteration is because of dynamic 
        topologies - if the particle wants its neighbors at a certain 
        iteration, make sure the neighbors are also on that iteration.
        """
        realneighbors = []
        for cand in messages:
            olditer = cand.iters
            cand.iters = particle.iters
            self.specex.set_neighborhood_rand(cand)
            if particle.id in self.specex.topology.iterneighbors(cand):
                realneighbors.append(cand)
            cand.iters = olditer
        return realneighbors

    def update_child_bests(self, particle, best, child):
        """Update the pbest and nbest of a child particle from its parent.
        
        In speculative execution this is necessary because the child 
        particle doesn't know if the value that it found is better than the 
        value the parent found at its position; it only updated its pbest in 
        relation to its previous pbestval.
        Also, it doesn't know the value of the function at the nbest position
        that it guessed.  It has to get that from its parent.
        """
        comparator = self.specex.function.comparator
        if Particle.isbetter(particle.pbestval, child.pbestval, comparator):
            child.pbestpos = particle.pbestpos
            child.pbestval = particle.pbestval
        if Particle.isbetter(best.pbestval, child.nbestval, comparator):
            child.nbestpos = best.pbestpos
            child.nbestval = best.pbestval


class ReproducePSO(_SpecMethod):
    """Reproduce standard PSO, in case you couldn't tell from the class name...
    
    Perform speculative execution in such a way that the original PSO is
    reproduced exactly, just two iterations at a time.  A lot of communication
    is required to make sure the behavior is exactly the same as the original
    PSO, and any good information from speculative particles is ignored except
    on the branch that standard PSO would have taken.
    """

    def setup(self, specex, pruner):
        if not isinstance(pruner, OneCompleteIteration):
            raise ValueError('ReproducePSO must have OneCompleteIteration as '
                    'its pruner!')
        super(ReproducePSO, self).setup(specex, pruner)

    def message_ids(self, particle):
        """Decide which particles need messages from this particle.

        Only Particles ever have to send messages three neighbors down.
        When you send to a neighbor's neighbor you need to increment the 
        iteration to make sure you're sending it to the right neighbor in the
        end.
        """
        messages = set()
        self.specex.set_neighborhood_rand(particle)
        for n in self.specex.topology.iterneighbors(particle):
            messages.add(n)
            ndummy = Dummy(n, particle.iters+1, particle.batches)
            self.specex.set_neighborhood_rand(ndummy)
            for n2 in self.specex.topology.iterneighbors(ndummy):
                messages.add(n2)
                if type(particle) == Particle:
                    n2dummy = Dummy(n2, particle.iters+2, particle.batches)
                    self.specex.set_neighborhood_rand(n2dummy)
                    for n3 in self.specex.topology.iterneighbors(n2dummy):
                        messages.add(n3)
        return messages

    def pick_child(self, particle, it1messages, children):
        """To find the correct branch that PSO would have taken, you need to 
        see whether or not the particle updated its pbest and find out which
        of the particle's neighbors was the new nbest.  Then update the child
        that matches that branch and return it.
        """
        comparator = self.specex.function.comparator
        neighbors = self.get_neighbors(particle, it1messages)
        # Look at the messages to see which branch you actually took
        best = self.specex.findbest(neighbors)
        if particle.isbetter(best.pbestval, particle.nbestval, comparator):
            nbestid = best.id
        else:
            nbestid = -1

        # If you updated your pbest, then your current value will be your
        # pbestval
        updatedpbest = (particle.pbestval == particle.value)

        # Look through the children and pick the child that corresponds to the
        # branch you took
        for child in children:
            if child.specpbest == updatedpbest and \
                child.specnbestid == nbestid:
                    newchild = child.make_real_particle()
                    self.update_child_bests(particle, best, newchild)
                    return newchild
        raise RuntimeError("Didn't find a child that matched the right branch!")

    def pick_neighbor_children(self, particle, it1messages, it2messages):
        """This function grabs the particle's neighbors from it1messages, 
        figures out which of their children is the correct one to take from
        it2messages, updates their pbest (done in pick_child), and returns them.
        """
        neighbors = self.get_neighbors(particle, it1messages)
        neighbor_children = []
        for neighbor in neighbors:
            children = []
            for child in it2messages:
                if child.id == neighbor.id:
                    children.append(child)
            realchild = self.pick_child(neighbor, it1messages, children)
            neighbor_children.append(realchild)
        return neighbor_children

    def update_neighbor_nbest(self, neighbors, it1messages, it2messages):
        comparator = self.specex.function.comparator
        for neighbor in neighbors:
            n = self.pick_neighbor_children(neighbor, it1messages, it2messages)
            best = self.specex.findbest(n)
            neighbor.nbest_cand(best.pbestpos, best.pbestval, comparator)


class PickBestChild(_SpecMethod):
    """
    """

class _Pruning(ParamObj):
    """Speculative Pruning
    
    You have an infinite tree of possible executions.  This class defines
    one method, generate_children, that decides how far to go down the 
    tree, and in what way.
    """

    def setup(self, specex):
        self.specex = specex

    def generate_children(self, particle, neighbors):
        """Given a particle and its neighbors, generate the particle's
        speculative children."""
        raise NotImplementedError


class OneCompleteIteration(_Pruning):

    def generate_children(self, particle, neighbors):
        """Given the particle and neighbors at iteration i, yield all possible
        speculative children for iteration i+1."""
        # Create children guessing that pbest was not updated
        child = SEParticle(particle, specpbest=False, specnbestid=-1)
        self.specex.set_motion_rand(child)
        self.specex.just_move(child)
        yield child
        for n in neighbors:
            if n.id != particle.id:
                child = SEParticle(particle, specpbest=False, specnbestid=n.id)
                child.nbestpos = n.pos
                self.specex.set_motion_rand(child)
                self.specex.just_move(child)
                yield child

        # Create children guessing that pbest was updated
        child = SEParticle(particle, specpbest=True, specnbestid=-1)
        child.pbestpos = particle.pos
        self.specex.set_motion_rand(child)
        self.specex.just_move(child)
        yield child
        for n in neighbors:
            child = SEParticle(particle, specpbest=True, specnbestid=n.id)
            child.nbestpos = n.pos
            child.pbestpos = particle.pos
            self.specex.set_motion_rand(child)
            self.specex.just_move(child)
            yield child

# vim: et sw=4 sts=4
