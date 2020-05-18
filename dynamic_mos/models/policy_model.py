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

    def __init__(self, robot_id, grid_map=None,
                 look_after_move=False):
        """FindAction can only be taken after LookAction"""
        self.robot_id = robot_id
        self.grid_map = grid_map

        self._look_after_move = look_after_move        
        if self._look_after_move:
            distance_cost = STEP_SIZE + Look.cost
        else:
            distance_cost = STEP_SIZE
        self.all_motion_actions = create_motion_actions(scheme="xy",
                                                         distance_cost=distance_cost)

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]
    
    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplementedError

    def _valid_actions(self, motions=None, can_find=False):
        find_action = set({Find}) if can_find else set({})
        if motions is None:
            motions = self.all_motion_actions
        if self._look_after_move:
            return motions | find_action
        else:
            return motions | {Look} | find_action

    def get_all_actions(self, state=None, history=None):
        """note: find can only happen after look."""
        can_find = True
        if not self._look_after_move:
            can_find = False
            if history is not None and len(history) > 1:
                # last action
                last_action = history[-1][0]
                if isinstance(last_action, LookAction):
                    can_find = True
        valid_motions = None        
        if state is not None and self.grid_map is not None:
            valid_motions =\
                self.grid_map.valid_motions(self.robot_id,
                                            state.pose(self.robot_id),
                                            self.all_motion_actions)
        return self._valid_actions(motions=valid_motions,
                                   can_find=can_find)
                
    def rollout(self, state, history):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]
    

# Preferred policy, action prior.    
class PreferredPolicyModel(PolicyModel):
    """The same with PolicyModel except there is a preferred rollout policypomdp_py.RolloutPolicy"""
    def __init__(self, action_prior, look_after_move=False):
        self.action_prior = action_prior
        super().__init__(self.action_prior.robot_id,
                         self.action_prior.grid_map,
                         look_after_move=look_after_move)
        self.action_prior.set_motion_actions(self.all_motion_actions)
        
    def rollout(self, state, history):
        # Obtain preference and returns the action in it.
        preferences = self.action_prior.get_preferred_actions(state, history)
        if len(preferences) > 0:
            return random.sample(preferences, 1)[0][0]
        else:
            return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

    
class DynamicMosActionPrior(pomdp_py.ActionPrior):
    def __init__(self, robot_id, grid_map, num_visits_init, val_init,
                 look_after_move=False):
        """level (int) used to set the level of 'manual'ness"""        
        self.robot_id = robot_id
        self.grid_map = grid_map
        self.all_motion_actions = None
        self.num_visits_init = num_visits_init
        self.val_init = val_init
        self._look_after_move = look_after_move
        self._level = level

    def set_motion_actions(self, motion_actions):
        self.all_motion_actions = motion_actions
        
    def get_preferred_actions(self, state, history):
        """Get preferred actions. This can be used by a rollout policy as well."""
        # Prefer actions that move the robot closer to any
        # undetected target object in the state. If
        # cannot move any closer, look. If the last
        # observation contains an unobserved object, then Find.
        if self.all_motion_actions is None:
            raise ValueError("Unable to get preferred actions because"\
                             "we don't know what motion actions there are.")
        robot_state = state.object_states[self.robot_id]

        if len(history) > 0:
            last_action, last_observation = history[-1]
            for objid in last_observation.objposes:
                if objid not in robot_state["objects_found"]\
                   and last_observation.for_obj(objid).pose != ObjectObservation.NULL:
                    # We last observed an object that was not found. Then Find.
                    return set({(FindAction(), self.num_visits_init, self.val_init)})

        if self._look_after_move:
            # No Look action; It's embedded in Move.
            preferences = set()
        else:
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
                                                    self.all_motion_actions))
                for next_robot_pose in neighbors:
                    if euclidean_dist(next_robot_pose, object_pose) < cur_dist:
                        preferences.add((neighbors[next_robot_pose],
                                             self.num_visits_init, self.val_init))
        return preferences
                
