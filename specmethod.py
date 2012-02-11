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
        self.check_compatibility(pruner)
        self.specex = specex
        self.pruner = pruner

    ##########################################################################
    # Methods that must be overridden

    def pick_child(self, particle, it1messages, children):
        """Returns particle at iteration 2 (minus nbest).

        particle is at iteration 1, children are all speculative children of
        particle (at iteration 2).  it1messages contain all messages that
        particle received in the Reduce phase.

        This function cannot modify the state of any of its arguments, or else
        very bad things happen.  The particle that is returned should be created
        from child.make_real_particle(), and that new particle's state should
        be updated, not the child message itself.
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

    def check_compatibility(self, pruner):
        """Some SpecMethods require certain pruners.  Check for compatibility.

        Raises an error if the pruner is incompatible with the SpecMethod,
        does nothing if the pruner is compatible."""
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


class ReproducePSO(_SpecMethod):
    """Reproduce standard PSO, in case you couldn't tell from the class name...

    Perform speculative execution in such a way that the original PSO is
    reproduced exactly, just two iterations at a time.  A lot of communication
    is required to make sure the behavior is exactly the same as the original
    PSO, and any good information from speculative particles is ignored except
    on the branch that standard PSO would have taken.
    """

    def setup(self, specex, pruner):
        super(ReproducePSO, self).setup(specex, pruner)

    def check_compatibility(self, pruner):
        if not isinstance(pruner, OneCompleteIteration):
            raise ValueError('ReproducePSO must have OneCompleteIteration as '
                    'its pruner!')

    def message_ids(self, particle):
        """Decide which particles need messages from this particle.

        Only Particles ever have to send messages three neighbors down.
        When you send to a neighbor's neighbor you need to increment the
        iteration to make sure you're sending it to the right neighbor in the
        end.
        """
        ids = set()
        self.specex.set_neighborhood_rand(particle)
        for n in self.specex.topology.iterneighbors(particle):
            ids.add(n)
            ndummy = Dummy(n, particle.iters+1)
            self.specex.set_neighborhood_rand(ndummy)
            for n2 in self.specex.topology.iterneighbors(ndummy):
                ids.add(n2)
                if type(particle) == Particle:
                    n2dummy = Dummy(n2, particle.iters+2)
                    self.specex.set_neighborhood_rand(n2dummy)
                    for n3 in self.specex.topology.iterneighbors(n2dummy):
                        ids.add(n3)
        return ids

    def pick_child(self, particle, it1messages, children):
        """To find the correct branch that PSO would have taken, you need to
        see whether or not the particle updated its pbest and find out which
        of the particle's neighbors was the new nbest.  Then update the child
        that matches that branch and return it.  Nothing that got passed into
        this function should have modified state at the end.
        """
        comparator = self.specex.function.comparator
        neighbors = self.get_neighbors(particle, it1messages)
        # Look at the messages to see which branch you actually took.
        # You only need an isbetter to figure out the branch, and then you
        # don't modify the state of the messages with an nbest_cand (as this
        # same method gets called with a message as 'particle' from
        # pick_neighbor_children).
        best_neighbor = self.specex.findbest(neighbors)
        if particle.isbetter(best_neighbor.pbestval, particle.nbestval,
                comparator):
            nbestid = best_neighbor.id
        else:
            nbestid = -1

        # If you updated your pbest, then your current value will be your
        # pbestval.  It's a floating point comparison, but the values would
        # have been set from each other, so it's safe.
        updatedpbest = (particle.pbestval == particle.value)

        # Look through the children and pick the child that corresponds to the
        # branch you took
        for child in children:
            if (child.specpbest == updatedpbest and
                    child.specnbestid == nbestid):
                # Make the child into a particle before modifying its state,
                # so that the messages never get modified.
                newchild = child.make_real_particle()
                self.update_child_bests(particle, best_neighbor, newchild)
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

    def update_neighbor_nbest(self, neighbors, it1messages, it2messages,
            it3=False):
        """Self explanatory?  I think so.  Update your neighbors' nbest with
        the messages that they received.
        """
        if it3 and self.pruner.no_neighbors:
            return
        comparator = self.specex.function.comparator
        for neighbor in neighbors:
            n = self.pick_neighbor_children(neighbor, it1messages, it2messages)
            best = self.specex.findbest(n)
            # Neighbors here are new particles, not messages, so you can, in
            # fact you must, modify their state with nbest_cand.
            neighbor.nbest_cand(best.pbestpos, best.pbestval, comparator)

    def update_child_bests(self, particle, best_neighbor, child):
        """Update the pbest and nbest of a child particle from its parent.

        In speculative execution this is necessary because the child
        particle doesn't know if the value that it found is better than the
        value the parent found at its position; it only updated its pbest in
        relation to its previous pbestval.
        Also, it doesn't know the value of the function at the nbest position
        that it guessed.  It has to get that from its parent.
        So, this function passes along the iteration 1 nbestval that was
        determined in pick_child and updates the iteration 2 pbest.  The
        child is at iteration 2 (minus nbest) after this method.

        Best is passed in separately from particle to be sure that the state
        of particle doesn't have to be changed in pick_child, because sometimes
        it's just a message that you really don't want to change.
        """
        comparator = self.specex.function.comparator
        # Passing along the nbestval from iteration 1
        if Particle.isbetter(best_neighbor.pbestval, child.nbestval,
                comparator):
            # In the ReproducePSO case this first line could be an assert, but
            # not in the PickBestChild case.
            child.nbestpos = best_neighbor.pbestpos
            child.nbestval = best_neighbor.pbestval
        # Set the pbest for iteration 2, and update the last branch
        if Particle.isbetter(particle.pbestval, child.pbestval, comparator):
            child.pbestpos = particle.pbestpos
            child.pbestval = particle.pbestval
            child.lastbranch[0] = False
        else:
            child.lastbranch[0] = True


class PickBestChild(ReproducePSO):
    """Use all information from speculative particles and pick the best child.

    ReproducePSO shows that speculative execution can be done such that PSO
    remains unchanged.  However, a lot of information from the speculative
    particles is wasted, even if it was better than the branch that standard
    PSO would have taken.  This method uses that information and always assumes
    that the correct speculative child is the one with the best value.

    The only place this method is different from ReproducePSO is in the
    pick_child method, so we inherit from it.

    We also allow this SpecMethod to be used with any pruner.  Using it with
    a pruner other than OneCompleteIteration could have interesting results.
    """

    def check_compatibility(self, pruner):
        pass

    def pick_child(self, particle, it1messages, children):
        """Instead of picking the branch that matches PSO, just take the branch
        that produced the best value.  If there are no speculative children for
        whatever reason, return the particle like in SocialPromotion.

        Nothing that got passed into this function should have modified state
        at the end.
        """
        if not children:
            newparticle = particle.copy()
            newparticle.iters += 1
            return newparticle
        # We still do this so that the child will have the correct nbestval.
        comparator = self.specex.function.comparator
        neighbors = self.get_neighbors(particle, it1messages)
        best = self.specex.findbest(neighbors)

        # Look through the children and pick the child that has the best value
        bestchild = None
        for child in children:
            if (bestchild is None) or comparator(child.value, bestchild.value):
                bestchild = child
        newchild = bestchild.make_real_particle()
        newchild.iters = particle.iters + 1
        newchild.lastbranch = [bestchild.specpbest, bestchild.specnbestid]
        self.update_child_bests(particle, best, newchild)
        return newchild


class SocialPromotion(ReproducePSO):
    """If the correct branch is not found, return the original particle.

    This SpecMethod is for use with incomplete pruners.  If a complete pruner
    is used, it results in exactly the same behavior as ReproducePSO.  If the
    pruning is incomplete, we look to see if the correct branch was one of the
    ones speculated about.  If not, we leave the particle an iteration behind
    everyone else.  That is equivalent to pulling the particle as it is one
    iteration forward, but the latter method makes dealing with topologies and
    random seeds a lot easier.
    """

    def check_compatibility(self, pruner):
        if isinstance(pruner, ManyIters):
            raise ValueError('SocialPromotion does not work when you go more '
                    'than one iteration ahead!')

    def pick_child(self, particle, it1messages, children):
        """We do the same here as in ReproducePSO, with one modification.  If
        the correct branch is not found, instead of throwing an error we just
        return the original particle, one iteration ahead.
        """
        comparator = self.specex.function.comparator
        neighbors = self.get_neighbors(particle, it1messages)
        # Look at the messages to see which branch you actually took.
        # You only need an isbetter to figure out the branch, and then you
        # don't modify the state of the messages with an nbest_cand (as this
        # same method gets called with a message as 'particle' from
        # pick_neighbor_children).
        best_neighbor = self.specex.findbest(neighbors)
        if particle.isbetter(best_neighbor.pbestval, particle.nbestval,
                comparator):
            nbestid = best_neighbor.id
        else:
            nbestid = -1

        # If you updated your pbest, then your current value will be your
        # pbestval.  It's a floating point comparison, but the values would
        # have been set from each other, so it's safe.
        updatedpbest = (particle.pbestval == particle.value)

        # Update the last branch that was taken, but only if you're a particle.
        # We don't really care about the branch messages took.  Also, this only
        # matters here if we didn't guess the right branch.  If we did, then
        # the child will get updated branch information later.
        if type(particle) == Particle:
            particle.lastbranch = [updatedpbest, nbestid]

        # Look through the children and pick the child that corresponds to the
        # branch you took
        for child in children:
            if (child.specpbest == updatedpbest and
                    child.specnbestid == nbestid):
                # Make the child into a particle before modifying its state,
                # so that the messages never get modified.
                newchild = child.make_real_particle()
                self.update_child_bests(particle, best_neighbor, newchild)
                return newchild

        # We didn't find the right child--our speculation was insufficient.
        # In that case, we redo the second iteration so we get the right
        # branch.  But to actually implement that, what we do is pull the
        # original particle ahead one iteration.
        newparticle = particle.copy()
        newparticle.iters += 1
        return newparticle


class _Pruner(ParamObj):
    """Speculative Pruner

    You have an infinite tree of possible executions.  This class defines
    one method, generate_children, that decides how far to go down the
    tree, and in what way.
    """

    def setup(self, specex):
        self.no_neighbors = False
        self.specex = specex

    def generate_children(self, particle, neighbors):
        """Given a particle and its neighbors, generate the particle's
        speculative children."""
        raise NotImplementedError


class OneCompleteIteration(_Pruner):

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


class NoNBestUpdate(_Pruner):

    def setup(self, specex):
        super(NoNBestUpdate, self).setup(specex)
        self.no_neighbors = True

    def generate_children(self, particle, neighbors):
        """Just speculate two children, the two where we assume we didn't get
        an nbest update.
        """
        # Create child guessing that pbest was not updated
        child = SEParticle(particle, specpbest=False, specnbestid=-1)
        self.specex.set_motion_rand(child)
        self.specex.just_move(child)
        yield child
        # And now guessing that pbest was updated
        child = SEParticle(particle, specpbest=True, specnbestid=-1)
        child.pbestpos = particle.pos
        self.specex.set_motion_rand(child)
        self.specex.just_move(child)
        yield child


class Stagnant(_Pruner):

    def generate_children(self, particle, neighbors):
        """Just one particle, for the stagnant case.
        """
        child = SEParticle(particle, specpbest=False, specnbestid=-1)
        self.specex.set_motion_rand(child)
        self.specex.just_move(child)
        yield child


class Stats(_Pruner):

    def generate_children(self, particle, neighbors):
        """Produce two children, one for no nbest, no pbest, and one for the
        last branch that was taken by this particle.
        """
        # Create child guessing that neither pbest nor nbest was updated
        child = SEParticle(particle, specpbest=False, specnbestid=-1)
        self.specex.set_motion_rand(child)
        self.specex.just_move(child)
        yield child
        # And now for the last branch that was taken
        if particle.lastbranch[0] == True:
            specpbest = True
        else:
            specpbest = False
        specnbestid = particle.lastbranch[1]
        child = SEParticle(particle, specpbest=specpbest,
                specnbestid=specnbestid)
        self.specex.set_motion_rand(child)
        self.specex.just_move(child)
        yield child


class LastBranch(_Pruner):

    def generate_children(self, particle, neighbors):
        """Produce just one child, corresponding to the last branch that was
        taken by this particle.
        """
        # And now for the last branch that was taken
        if particle.lastbranch[0] == True:
            specpbest = True
        else:
            specpbest = False
        specnbestid = particle.lastbranch[1]
        child = SEParticle(particle, specpbest=specpbest,
                specnbestid=specnbestid)
        self.specex.set_motion_rand(child)
        self.specex.just_move(child)
        yield child


class TokenBasedOneIter(_Pruner):
    """Produce however many children you have tokens for.

    Start with no nbest, no pbest.  If you have two tokens, move to pbest, but
    no nbest.  Then take the last branch.  Then just start randomly guessing.
    """

    def generate_children(self, particle, neighbors):
        if particle.tokens > 0:
            children = []
            # Create child guessing that neither pbest nor nbest was updated
            child = SEParticle(particle, specpbest=False, specnbestid=-1)
            self.specex.set_motion_rand(child)
            self.specex.just_move(child)
            children.append((False, -1))
            yield child
        if particle.tokens > 1:
            # And now guessing that pbest was updated
            child = SEParticle(particle, specpbest=True, specnbestid=-1)
            child.pbestpos = particle.pos
            self.specex.set_motion_rand(child)
            self.specex.just_move(child)
            children.append((True, -1))
            yield child
        if particle.tokens > 2:
            # And now for the last branch that was taken
            if particle.lastbranch[0] == True:
                specpbest = True
            else:
                specpbest = False
            specnbestid = particle.lastbranch[1]
            if specnbestid != -1:
                child = SEParticle(particle, specpbest=specpbest,
                        specnbestid=specnbestid)
                self.specex.set_motion_rand(child)
                self.specex.just_move(child)
            else:
                self.specex.set_neighborhood_rand(particle)
                specnbestid = particle.rand.randrange(len(neighbors))
                child = SEParticle(particle, specpbest=specpbest,
                        specnbestid=specnbestid)
                self.specex.set_motion_rand(child)
                self.specex.just_move(child)
            children.append((specpbest, specnbestid))
            yield child
        if particle.tokens > 3:
            for i in range(particle.tokens-3):
                tries = 0
                failed = False
                while (specpbest, specnbestid) in children:
                    tries += 1
                    if tries > 10:
                        failed = True
                        break
                    if particle.rand.randrange(2):
                        specpbest = True
                    else:
                        specpbest = False
                    specnbestid = particle.rand.randrange(len(neighbors))
                if failed:
                    break
                child = SEParticle(particle, specpbest=specpbest,
                        specnbestid=specnbestid)
                self.specex.set_motion_rand(child)
                self.specex.just_move(child)
                children.append((specpbest, specnbestid))
                yield child


class ManyIters(_Pruner):
    """A pruning super-class for all pruners that go more than one iteration
    ahead.

    This is useful so that speculative methods that aren't capable of going
    more than one iteration ahead can throw an error on just one class, instead
    of having to know about all of the possible pruners.
    """

class ManyItersSevenEvals(ManyIters):
    """Use the same number of evaluations as OneCompleteIteration, but go many
    iterations ahead.

    We do the same evaluations as TokenBasedManyIters, but we don't look at
    tokens.  All of the particles do the same number of evaluations.
    """

    def setup(self, specex):
        super(ManyItersSevenEvals, self).setup(specex)
        self.no_neighbors = True

    def generate_children(self, particle, neighbors):
        # Iteration 2, NN
        child = SEParticle(particle, specpbest=False, specnbestid=-1)
        self.specex.just_move(child)
        yield child
        # Iteration 2, YN
        child = SEParticle(particle, specpbest=True, specnbestid=-1)
        child.pbestpos = particle.pos
        self.specex.just_move(child)
        yield child
        # Iteration 3, NN-NN
        child = SEParticle(particle, specpbest=False, specnbestid=-1)
        self.specex.just_move(child)
        child = SEParticle(child, specpbest=False, specnbestid=-1)
        self.specex.just_move(child)
        yield child
        # Iteration 3, NN-YN
        child = SEParticle(particle, specpbest=False, specnbestid=-1)
        self.specex.just_move(child)
        child.pbestpos = child.pos
        child = SEParticle(child, specpbest=True, specnbestid=-1)
        self.specex.just_move(child)
        yield child
        # Iteration 3, YN-NN
        child = SEParticle(particle, specpbest=True, specnbestid=-1)
        child.pbestpos = particle.pos
        self.specex.just_move(child)
        child = SEParticle(child, specpbest=False, specnbestid=-1)
        self.specex.just_move(child)
        yield child
        # Iteration 3, YN-YN
        child = SEParticle(particle, specpbest=True, specnbestid=-1)
        child.pbestpos = particle.pos
        self.specex.just_move(child)
        child.pbestpos = child.pos
        child = SEParticle(child, specpbest=True, specnbestid=-1)
        self.specex.just_move(child)
        yield child
        # Iteration 4, NN-NN-NN
        child = SEParticle(particle, specpbest=False, specnbestid=-1)
        self.specex.just_move(child)
        child = SEParticle(child, specpbest=False, specnbestid=-1)
        self.specex.just_move(child)
        child = SEParticle(child, specpbest=False, specnbestid=-1)
        self.specex.just_move(child)
        yield child


class TokenBasedManyIters(ManyIters):
    """Produce however many children you have tokens for.

    Start with no nbest, no pbest.  If you have two tokens, move to pbest, but
    no nbest.  Then go more than one iteration ahead, the same branches.
    """

    def generate_children(self, particle, neighbors):
        if particle.tokens > 0:
            # Iteration 2, NN
            child = SEParticle(particle, specpbest=False, specnbestid=-1)
            self.specex.just_move(child)
            yield child
        if particle.tokens > 1:
            # Iteration 2, YN
            child = SEParticle(particle, specpbest=True, specnbestid=-1)
            child.pbestpos = particle.pos
            self.specex.just_move(child)
            yield child
        if particle.tokens > 2:
            # Iteration 3, NN-NN
            child = SEParticle(particle, specpbest=False, specnbestid=-1)
            self.specex.just_move(child)
            child = SEParticle(child, specpbest=False, specnbestid=-1)
            self.specex.just_move(child)
            yield child
        if particle.tokens > 3:
            # Iteration 3, NN-YN
            child = SEParticle(particle, specpbest=False, specnbestid=-1)
            self.specex.just_move(child)
            child.pbestpos = child.pos
            child = SEParticle(child, specpbest=True, specnbestid=-1)
            self.specex.just_move(child)
            yield child
        if particle.tokens > 4:
            # Iteration 3, YN-NN
            child = SEParticle(particle, specpbest=True, specnbestid=-1)
            child.pbestpos = particle.pos
            self.specex.just_move(child)
            child = SEParticle(child, specpbest=False, specnbestid=-1)
            self.specex.just_move(child)
            yield child
        if particle.tokens > 5:
            # Iteration 3, YN-YN
            child = SEParticle(particle, specpbest=True, specnbestid=-1)
            child.pbestpos = particle.pos
            self.specex.just_move(child)
            child.pbestpos = child.pos
            child = SEParticle(child, specpbest=True, specnbestid=-1)
            self.specex.just_move(child)
            yield child
        if particle.tokens > 6:
            # Iteration 4, NN-NN-NN
            child = SEParticle(particle, specpbest=False, specnbestid=-1)
            self.specex.just_move(child)
            child = SEParticle(child, specpbest=False, specnbestid=-1)
            self.specex.just_move(child)
            child = SEParticle(child, specpbest=False, specnbestid=-1)
            self.specex.just_move(child)
            yield child
        if particle.tokens > 7:
            # Iteration 4, NN-NN-YN
            child = SEParticle(particle, specpbest=False, specnbestid=-1)
            self.specex.just_move(child)
            child = SEParticle(child, specpbest=False, specnbestid=-1)
            self.specex.just_move(child)
            child.pbestpos = child.pos
            child = SEParticle(child, specpbest=True, specnbestid=-1)
            self.specex.just_move(child)
            yield child


class TokenManyItersOnlyPbest(ManyIters):
    """Produce however many children you have tokens for down the YN branch.
    """

    def generate_children(self, particle, neighbors):
        if particle.tokens > 0:
            child = SEParticle(particle, specpbest=True, specnbestid=-1)
            child.pbestpos = particle.pos
            self.specex.just_move(child)
            yield child
            i = 1
            while i < particle.tokens:
                i += 1
                child = SEParticle(child, specpbest=True, specnbestid=-1)
                child.pbestpos = child.pos
                self.specex.just_move(child)
                yield child


class TokenManyItersOnlyNbest(ManyIters):
    """Produce however many children you have tokens for down the YN branch.
    """

    def generate_children(self, particle, neighbors):
        if particle.tokens > 0:
            child = SEParticle(particle, specpbest=False, specnbestid=-1)
            self.specex.just_move(child)
            yield child
            i = 1
            while i < particle.tokens:
                i += 1
                child = SEParticle(child, specpbest=False, specnbestid=-1)
                self.specex.just_move(child)
                yield child


# vim: et sw=4 sts=4
