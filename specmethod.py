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
        """Given a particle, its neighbors, and its children, decide which 
        child to return."""
        raise NotImplementedError

    def pick_neighbor_children(self, particle, it1messages, it2messages):
        """Given a set of neighbors and their children and neighbors, return a
        set of children, one for each neighbor.  In some cases having the
        particle is also necessary to determine which neighbors are actually
        neighbors."""
        raise NotImplementedError

    def update_neighbor_nbest(self, neighbors, it1messages, it2messages):
        """Given a set of neighbors with everything but their nbest and
        messages for those neighbors, update the neighbors' nbest.
        Some speculative methods skip this step to save on messages that
        need to be passed, so the function would be empty.
        """
        raise NotImplementedError

    def itermessages(self, particle):
        """Given a particle, return the set of particles to whom it must
        send messages."""
        raise NotImplementedError

    ##########################################################################
    # Methods that probably should not be overridden.  

    def generate_children(self, particle, neighbors):
        """Yield the speculative children of a particle given it and its
        neighbors.  Often uses an external pruning class."""
        return self.pruner.generate_children(particle, neighbors)

    def get_real_neighbors(self, particle, messages):
        """This assumes a non-symmetric topology.  It is a bit less work if the
        topology is symmetric, but this works in general.  Also, messing with
        the neighbors' iteration is because of dynamic topologies - if the
        particle wants its neighbors at a certain iteration, make sure the
        neighbors are also on that iteration."""
        realneighbors = []
        for cand in messages:
            olditer = cand.iters
            if cand.iters != particle.iters:
                cand.iters = particle.iters
            self.specex.set_neighborhood_rand(cand, swarmid=0)
            if particle.id in self.specex.topology.iterneighbors(cand):
                realneighbors.append(cand)
            cand.iters = olditer
        return realneighbors

    def update_child_bests(self, particle, child):
        """In speculative execution this is necessary because the child 
        particle doesn't know if the value that it found is better than the 
        value the parent found at its position, it only updated its pbest in 
        relation to its previous pbestval.
        Also, it doesn't know the value of the function at the nbest position
        that it guessed.  It has to get that from its parent.
        """
        comparator = self.specex.function.comparator
        if Particle.isbetter(particle.pbestval, child.pbestval, comparator):
            child.pbestpos = particle.pbestpos
            child.pbestval = particle.pbestval
        if Particle.isbetter(particle.nbestval, child.nbestval, comparator):
            assert(child.nbestpos == particle.nbestpos)
            child.nbestval = particle.nbestval


class ReproducePSO(_SpecMethod):
    """Perform speculative execution in such a way that the original PSO is
    reproduced exactly, just two iterations at a time.  A lot of communication
    is required to make sure the behavior is exactly the same as the original
    PSO, and any good information from speculative particles is ignored except
    on the branch that standard PSO would have taken.
    """

    def setup(self, specex, pruner):
        if not isinstance(pruner, OneCompleteIteration):
            raise ValueError('ReproducePSO must have OneCompleteIteration as '
                    'its pruner!')
        self.pruner = pruner
        self.specex = specex

    def itermessages(self, particle):
        neighbors = set()
        if type(particle == Particle):
            self.specex.set_neighborhood_rand(particle, swarmid=0)
            for n in self.specex.topology.iterneighbors(particle):
                neighbors.add(n)
        particle.iters += 1
        for n in self.specex.topology.iterneighbors(particle):
            neighbors.add(n)
            if type(particle == Particle):
                ndummy = Dummy(n, particle.iters, particle.batches)
                self.specex.set_neighborhood_rand(ndummy, swarmid=0)
                for n2 in self.specex.topology.iterneighbors(ndummy):
                    neighbors.add(n2)
        particle.iters += 1
        for n in self.specex.topology.iterneighbors(particle):
            neighbors.add(n)
            ndummy = Dummy(n, particle.iters, particle.batches)
            self.specex.set_neighborhood_rand(ndummy, swarmid=0)
            for n2 in self.specex.topology.iterneighbors(ndummy):
                neighbors.add(n2)
                if type(particle == Particle):
                    n2dummy = Dummy(n2, particle.iters, particle.batches)
                    self.specex.set_neighborhood_rand(n2dummy, swarmid=0)
                    for n3 in self.specex.topology.iterneighbors(n2dummy):
                        neighbors.add(n3)
        particle.iters -= 2
        return neighbors

    def pick_child(self, particle, it1messages, children):
        """To find the correct branch the PSO would have taken, you need to 
        see whether or not the particle updated its pbest and find out which
        of the particle's neighbors was the new nbest.  Then update the child
        that matches that branch and return it.
        """
        comparator = self.specex.function.comparator
        neighbors = self.get_real_neighbors(particle, it1messages)
        # Look at the messages to see which branch you actually took
        best = self.specex.findbest(neighbors, comparator)
        particle.nbest_cand(best.pbestpos, best.pbestval, comparator)

        # If you updated your pbest, then your current value will be your
        # pbestval
        updatedpbest = particle.pbestval == particle.value

        if particle.nbestval == best.pbestval:
            bestid = best.id
        else:
            bestid = -1

        # Look through the children and pick the child that corresponds to the
        # branch you took
        for child in children:
            if child.specpbest == updatedpbest and \
                child.specnbestid == bestid:
                    self.update_child_bests(particle, child)
                    return child.make_real_particle()
        raise RuntimeError("Didn't find a child that matched the right branch!")

    def pick_neighbor_children(self, particle, it1messages, it2messages):
        """This function grabs the particle's neighbors from it1messages, 
        figures out which of their children is the correct one to take from
        it2messages, updates their pbest (done in pick_child), and returns them.
        """
        neighbors = self.get_real_neighbors(particle, it1messages)
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
            best = self.specex.findbest(n, comparator)
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
        self.specex.set_motion_rand(child, swarmid=0)
        self.specex.just_move(child)
        yield child
        for n in neighbors:
            if n.id != particle.id:
                child = SEParticle(particle, specpbest=False, specnbestid=n.id)
                child.nbestpos = n.pos
                self.specex.set_motion_rand(child, swarmid=0)
                self.specex.just_move(child)
                yield child

        # Create children guessing that pbest was updated
        child = SEParticle(particle, specpbest=True, specnbestid=-1)
        child.pbestpos = particle.pos
        self.specex.set_motion_rand(child, swarmid=0)
        self.specex.just_move(child)
        yield child
        for n in neighbors:
            child = SEParticle(particle, specpbest=True, specnbestid=n.id)
            child.nbestpos = n.pos
            child.pbestpos = particle.pos
            self.specex.set_motion_rand(child, swarmid=0)
            self.specex.just_move(child)
            yield child

# vim: et sw=4 sts=4
