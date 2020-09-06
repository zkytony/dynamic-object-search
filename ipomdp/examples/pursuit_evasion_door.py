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
from ipomdp.examples.domain import *
    
#------Transition, Observation, Reward, Policy models------#
# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    """This problem is small enough for the probabilities to be directly given
            externally"""
    def __init__(self, role):
        self.role = role

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


# class ObservationModel(pomdp_py.ObservationModel):
#     """This problem is small enough for the probabilities to be directly given
#     externally"""
#     def __init__(self, role):
#         self.role = role

#     def probability(self, observation, next_state, action, normalized=False, **kwargs):
#         """
#         observation (observation)
#         next_state (JointState)
#         action (JointAction)
#         """        
#         if observation == "noise":
#             if next_state.pstate.side == next_state.estate.side:
#                 return 0.85  # both agents are on the same side
#             else:
#                 return 0.15
#         elif observation == "silence":
#             if next_state


        
#         return self._probs[next_state][action][observation]

#     def sample(self, next_state, action, normalized=False, **kwargs):
#         return self.get_distribution(next_state, action).random()

#     def argmax(self, next_state, action, normalized=False, **kwargs):
#         """Returns the most likely observation"""
#         return max(self._probs[next_state][action], key=self._probs[next_state][action].get)

    

# # Reward Model
# class RewardModel(pomdp_py.RewardModel):
#     def __init__(self, scale=1):
#         self._scale = scale
#     def _reward_func(self, state, action):
#         reward = 0
#         if action == "open-left":
#             if state== "tiger-right":
#                 reward += 10 * self._scale
#             else:
#                 reward -= 100 * self._scale
#         elif action == "open-right":
#             if state== "tiger-left":
#                 reward += 10 * self._scale
#             else:
#                 reward -= 100 * self._scale
#         elif action == "listen":
#             reward -= 1 * self._scale
#         return reward
    
#     def sample(self, state, action, next_state, normalized=False, **kwargs):
#         # deterministic
#         return self.argmax(state, action, next_state, normalized=normalized, **kwargs)

#     def argmax(self, state, action, next_state, normalized=False, **kwargs):
#         """Returns the most likely reward"""
#         return self._reward_func(state, action)

# # Policy Model
# class PolicyModel(pomdp_py.RandomRollout):
#     """This is an extremely dumb policy model; To keep consistent
#     with the framework."""
#     def sample(self, state, normalized=False, **kwargs):
#         return self.get_all_actions().random()
    
#     def get_all_actions(self, **kwargs):
#         return ACTIONS

if __name__ == "__main__":
    pstate = PState("left-door")
    estate = EState("right-door")
    state = JointState(pstate, estate)

    paction = PAction("move")
    eaction = EAction("stay")
    action = JointAction(paction, eaction)

    Tp = TransitionModel("pursuer")
    Te = TransitionModel("evader")

    print(Tp.sample(state, action))
    print(Te.sample(state, action))
