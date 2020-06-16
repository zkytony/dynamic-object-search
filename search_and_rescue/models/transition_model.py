import pomdp_py
import copy
from search_and_rescue.env.action import *
from search_and_rescue.env.state import *

class StaticTransitionModel(pomdp_py.TransitionModel):
    """This model assumes the object is static."""
    def __init__(self, objid, epsilon=1e-9):
        self._objid = objid
        self._epsilon = epsilon

    def probability(self, next_object_state, state, action):
        if next_object_state != state.object_states[next_object_state['id']]:
            return self._epsilon
        else:
            return 1.0 - self._epsilon
    
    def sample(self, state, action):
        """Returns next_object_state"""
        return self.argmax(state, action)
    
    def argmax(self, state, action):
        """Returns the most likely next object_state"""
        return copy.deepcopy(state.object_states[self._objid])

class DynamicTransitionModel(pomdp_py.TransitionModel):
    def __init__(self, objid, motion_policy, epsilon=1e-9,
                 is_agent=False, look_after_move=False):
        """
        objid (int) The id of the object that this dynamics model is describing.
                    It may or may not be the agent itself.
        is_agent (bool) True if this model describes the dynamics of the agent itself.
                        That means, the actions will be taken by this agent.
        """
        self._objid = objid
        self._motion_policy = motion_policy
        self._epsilon = epsilon
        self._is_agent = is_agent
        self._look_after_move = look_after_move
        
    def probability(self, next_object_state, state, action):
        if self._is_agent:
            return self._motion_policy.probability(next_object_state,
                                                   state, action)
        else:
            return self._motion_policy.probability(next_object_state,
                                                   state)

    def sample(self, state, action, argmax=False):
        if isinstance(action, MotionAction):
            if argmax:
                sample_func = self._motion_policy.argmax
            else:
                sample_func = self._motion_policy.random

            if self._is_agent:
                next_object_state = sample_func(state, action)
            else:
                # The action taken by the agent is not relevant to sample the next
                # position of this dynamic object which did not take the action.
                # AT LEAST FOR NOW.
                next_object_state = sample_func(state)
        else:
            next_object_state = copy.deepcopy(state.object_states[self._objid])

        if self._look_after_move\
           or (isinstance(action, LookAction) or isinstance(action, FindAction)):
            # Camera is turned on; The robot is looking
            next_object_state["camera_on"] = True
        else:
            # Camera is turned off
            next_object_state["camera_on"] = False
        return next_object_state
    
    
class JointTransitionModel(pomdp_py.OOTransitionModel):
    def __init__(self,
                 agent_id,
                 static_object_ids,
                 dynamic_object_ids,
                 motion_policies,
                 look_after_move=False):
        self.object_ids = {"static": static_object_ids,
                           "dynamic": dynamic_object_ids}
        transition_models = {}
        for objid in static_object_ids:
            transition_models[objid] = StaticTransitionModel(objid)
        for objid in dynamic_object_ids:
            transition_models[objid] = DynamicTransitionModel(objid, motion_policies[objid],
                                                              look_after_move=look_after_move,
                                                              is_agent=objid == agent_id)
        super().__init__(transition_models)
        
    def motion_policy(self, objid):
        return self.dynamic_object_motion_policies.get(objid, None)
        
    def sample(self, state, action, **kwargs):
        oostate = pomdp_py.OOTransitionModel.sample(self, state, action, **kwargs)
        return JointState(oostate.object_states)

    def argmax(self, state, action, normalized=False, **kwargs):
        oostate = pomdp_py.OOTransitionModel.argmax(self, state, action, **kwargs)
        return JointState(oostate.object_states)


def unittest():
    from search_and_rescue.models.motion_policy import BasicMotionPolicy, AdversarialPolicy
    from search_and_rescue.models.grid_map import GridMap
    grid_map = GridMap(10, 10,
                       {0: (2,3), 5:(4,9)})
    motion_actions = create_motion_actions(can_stay=True)
    t = JointTransitionModel(3,
                             {0,5}, {3,4},
                             {3: BasicMotionPolicy(3, grid_map, motion_actions),
                              4: AdversarialPolicy(4, 3, grid_map, 3, motion_actions=motion_actions)})
    state = JointState({3: VictimState(3, (1,0), False),
                        4: SearcherState(4, (4,4), (), True),
                        0: ObstacleState(0, (2,3)),
                        5: ObstacleState(5, (4,9))})
    action = MoveEast
    next_state = JointState({3: VictimState(3, (2,0), True),
                             4: SearcherState(4, (4,4), (), True),
                             0: ObstacleState(0, (2,3)),
                             5: ObstacleState(5, (4,9))})
    print(t.probability(next_state, state, action))
    print("---OBJ 3 Move east---")
    print(t.sample(state, action))
    print("---OBJ 3 Look---")
    print(t.sample(state, Look))    

if __name__ == '__main__':
    unittest()
    
