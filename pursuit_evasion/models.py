# This is a simple two-agent domain similar to the tiger domain.
#
# There are two doors, and one person is on one side, and the other
# on the other side.
#
# Each person is (states) either behind the left door, or the right door.
#
# They can both (actions):
#
# - move: move to the other door
# - stay (listen): don't move
# - open-door: open the door in front
#
# (observations) If they stay, they can hear noise, when the other person is at
# the same door as them, or silence if not.
#
# (reward) There is a pursuer and an evader. If when the door is open, and
# both agents are met face to face, then the pursuer wins. Otherwise, the
# evader wins. Whenever a door is open, the game ends.
from pursuit_evasion.domain import *
import random
    
#------Transition, Observation, Reward, Policy models------#
# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    """This problem is small enough for the probabilities to be directly given
            externally"""
    def probability(self, next_state, state, action, normalized=False, **kwargs):
        """
        next_state (JointState)
        action (JointAction)
        state (JointState)
        """
        for ai in action.tolist():
            si = state.for_role(ai.role)
            ai = action.for_role(ai.role)        
            spi = next_state.for_role(ai.role)
            if ai.name == "move":
                if si.side == spi.side:
                    return 0.0
            else:
                if si.side != spi.side:
                    return 0.0
        return 1.0
            
    def sample(self, state, action, normalized=False, **kwargs):
        next_state = state.copy()
        for ai in action.tolist():
            si = next_state.for_role(ai.role)        
            ai = action.for_role(ai.role)
            next_side = si.side
            if ai.name == "move":
                next_side = other_side(si.side)
            next_state.set_role(ai.role, si.__class__(next_side))
        return next_state


class ObservationModel(pomdp_py.ObservationModel):
    """
    Let i be the "self" agent and j be the "other" agent.
    If i moves or opens door, there is no observation.
    If i stays, and j is at the same door, there is "noise-near"
    If i stays, and j is at the different same door, there is "noise-far"
    The observation model is only for a single agent.
    """
    def __init__(self, role, certainty=0.85):
        self.role = role
        self.certainty = certainty

    def probability(self, observation, next_state, action, normalized=False, **kwargs):
        """
        observation (observation)
        next_state (JointState)
        action (JointAction)
        """
        ai = action.for_role(self.role)
        si = state.for_role(self.role)
        sj = state.for_role(other_role(self.role))        
        if ai.name == "stay":
            # Can hear noise
            if si.side == sj.side:
                # both agents on the same side
                if observation.name == "noise-near":
                    return self.certainty
                else:
                    return 1.0 - self.certainty
            else:
                # both agents on different side                
                if observation.name == "noise-far":
                    return self.certainty
                else:
                    return 1.0 - self.certainty
        return 0.5  # no information otherwise

    def sample(self, next_state, action, normalized=False, **kwargs):
        ai = action.for_role(self.role)
        si = state.for_role(self.role)
        sj = state.for_role(other_role(self.role))                
        if ai.name == "stay":
            # Can hear noise
            if si.side == sj.side:
                if random.uniform(0,1) > self.certainty:
                    return Observation("noise-near")
                else:
                    return Observation("noise-far")
            else:
                if random.uniform(0,1) > self.certainty:
                    return Observation("noise-far")
                else:
                    return Observation("noise-near")                

        if random.uniform(0,1) > 0.5:
            return Observation("noise-near")
        else:
            return Observation("noise-far")

# Reward Model
class RewardModel(pomdp_py.RewardModel):
    """
    If any one opens door and both agents appear
    on the same side, then there is a reward or penalty
    """
    def __init__(self, role):
        self.role = role
        
    def _reward_func(self, state, action):
        ai = action.for_role(self.role)
        aj = action.for_role(other_role(self.role))
        if ai.name == "open-door"\
           or aj.name == "open-door":
            si = state.for_role(self.role)
            sj = state.for_role(other_role(self.role))
            if si.side == sj.side:
                if self.role == "pursuer":
                    return 100
                else:
                    return -100
            else:
                if self.role == "pursuer":
                    return -100
                else:
                    return 100
        return -1
    
    def sample(self, state, action, next_state, normalized=False, **kwargs):
        # deterministic
        return self.argmax(state, action, next_state, normalized=normalized, **kwargs)

    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        return self._reward_func(state, action)

# Policy Model
class PolicyModel(pomdp_py.RandomRollout):
    """A random model"""
    def __init__(self, role):
        self.role = role
        if self.role == "pursuer":
            self.actions = [PAction("move"), PAction("stay"), PAction("open-door")]
        else:
            self.actions = [EAction("move"), EAction("stay"), EAction("open-door")]            
    
    def sample(self, state, **kwargs):
        return random.sample(self.get_all_actions(**kwargs), 1)[0]    
    
    def get_all_actions(self, **kwargs):
        return self.actions
    
    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]
    
    
if __name__ == "__main__":
    pstate = PState("left-door")
    estate = EState("right-door")
    state = JointState(pstate, estate)

    paction = PAction("move")
    eaction = EAction("move")
    action = JointAction(paction, eaction)

    Tp = TransitionModel("pursuer")
    Te = TransitionModel("evader")

    print(Tp.sample(state, action))
    print(Te.sample(state, action))

    Op = ObservationModel("pursuer")
    observation = Op.sample(Tp.sample(state, action),
                            action)
    print(action)
    print(observation)

    Rp = RewardModel("pursuer")
    print(Rp.sample(state, action, Tp.sample(state, action)))

    Pp = PolicyModel("pursuer")
    print(Pp.sample(state))
