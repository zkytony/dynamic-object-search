"""Policy model for 2D Multi-Object Search domain. 
It is optional for the agent to be equipped with an occupancy
grid map of the environment.
"""

import pomdp_py
import random
from ..domain.state import *
from ..domain.action import *
from ..domain.observation import *
from ..utils import *

class PolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model. All actions are possible at any state."""

    def __init__(self, robot_id, grid_map=None):
        """FindAction can only be taken after LookAction"""
        self.robot_id = robot_id
        self.grid_map = grid_map

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]
    
    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplementedError

    def get_all_actions(self, state=None, history=None):
        """note: find can only happen after look."""
        can_find = False
        if history is not None and len(history) > 1:
            # last action
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                can_find = True
        find_action = set({Find}) if can_find else set({})
        if state is None:
            return ALL_MOTION_ACTIONS | {Look} | find_action
        else:
            if self.grid_map is not None:
                valid_motions =\
                    self.grid_map.valid_motions(self.robot_id,
                                                 state.pose(self.robot_id),
                                                 ALL_MOTION_ACTIONS)
                return valid_motions | {Look} | find_action
            else:
                return ALL_MOTION_ACTIONS | {Look} | find_action

    def rollout(self, state, history):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

    def all_motion_actions(self):
        return ALL_MOTION_ACTIONS

    
class PreferredPolicyModel(PolicyModel):
    """The same with PolicyModel except there is a preferred rollout policypomdp_py.RolloutPolicy"""
    def __init__(self, action_prior):
        self.action_prior = action_prior
        super().__init__(self.action_prior.robot_id,
                         self.action_prior.grid_map)
        
    def rollout(self, state, history):
        # Obtain preference and returns the action in it.
        return random.sample(self.action_prior.get_preferred_actions(state, history), 1)[0][0]

    
class DynamicMosActionPrior(pomdp_py.ActionPrior):

    def __init__(self, robot_id, grid_map, num_visits_init, val_init):
        self.robot_id = robot_id
        self.grid_map = grid_map
        self.num_visits_init = num_visits_init
        self.val_init = val_init
        
    def get_preferred_actions(self, state, history):
        """Get preferred actions. This can be used by a rollout policy as well."""
        # Prefer actions that move the robot closer to any
        # undetected target object in the state. If
        # cannot move any closer, look. If the last
        # observation contains an unobserved object, then Find.
        robot_state = state.object_states[self.robot_id]

        if len(history) > 0:
            last_action, last_observation = history[-1]
            for objid in last_observation.objposes:
                if objid not in robot_state["objects_found"]\
                   and last_observation.for_obj(objid).pose != ObjectObservation.NULL:
                    # We last observed an object that was not found. Then Find.
                    return set({(FindAction(), self.num_visits_init, self.val_init)})

        # Always give preference to Look
        preferences = set({(LookAction(), self.num_visits_init, self.val_init)})
        for objid in state.object_states:
            if objid != self.robot_id and objid not in robot_state.objects_found:
                object_pose = state.pose(objid)
                cur_dist = euclidean_dist(robot_state.pose, object_pose)
                neighbors =\
                    self.grid_map.get_neighbors(
                        robot_state.pose,
                        self.grid_map.valid_motions(self.robot_id,
                                                    robot_state.pose,
                                                    ALL_MOTION_ACTIONS))
                for next_robot_pose in neighbors:
                    if euclidean_dist(next_robot_pose, object_pose) < cur_dist:
                        preferences.add((neighbors[next_robot_pose],
                                         self.num_visits_init, self.val_init))
        return preferences
                
            
