import pomdp_py
import random
from search_and_rescue.env.observation import *
from search_and_rescue.env.state import *
from search_and_rescue.env.action import *
from search_and_rescue.utils import *

class PolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model. All actions are possible at any state."""

    def __init__(self, robot_id, motion_policy=None,
                 look_after_move=False):
        """FindAction can only be taken after LookAction"""
        self.robot_id = robot_id
        self.motion_policy = motion_policy
        self._look_after_move = look_after_move
        if self._look_after_move:
            distance_cost = STEP_SIZE + Look.cost
        else:
            distance_cost = STEP_SIZE
        if self.motion_policy is not None:
            self.all_motion_actions = self.motion_policy.motion_actions
        else:
            self.all_motion_actions = create_motion_actions(distance_cost=distance_cost)

    def sample(self, state, **kwargs):
        return random.sample(self.get_all_actions(state=state, **kwargs), 1)[0]
    
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
        if state is not None and self.motion_policy is not None:
            valid_motions =\
                self.motion_policy.valid_motions(self.robot_id,
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
                         self.action_prior.motion_policy,
                         look_after_move=look_after_move)
        
    def rollout(self, state, history):
        # Obtain preference and returns the action in it.
        preferences = self.action_prior.get_preferred_actions(state, history)
        if len(preferences) > 0:
            return random.sample(preferences, 1)[0][0]
        else:
            return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

    
class GreedyActionPrior(pomdp_py.ActionPrior):
    def __init__(self, robot_id, motion_policy, num_visits_init, val_init,
                 look_after_move=False):
        self.robot_id = robot_id
        self.motion_policy = motion_policy
        self.num_visits_init = num_visits_init
        self.val_init = val_init
        self._look_after_move = look_after_move

    def get_preferred_actions(self, state, history):
        """Get preferred actions. This can be used by a rollout policy as well."""
        # Prefer actions that move the robot closer to any
        # undetected target object in the state. If
        # cannot move any closer, look. If the last
        # observation contains an unobserved object, then Find.
        if self.motion_policy.motion_actions is None:
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
            if hasattr(robot_state, "objects_found") and objid in robot_state.objects_found:
                # Object has been found; So no need to prefer actions for this object.
                continue
            if objid != self.robot_id:
                object_pose = state.pose(objid)
                cur_dist = euclidean_dist(robot_state.pose, object_pose)
                neighbors = self.motion_policy.get_neighbors(robot_state.pose,
                                                     self.motion_policy.motion_actions)
                for next_robot_pose in neighbors:
                    motion_action = neighbors[next_robot_pose]
                    next_dist = euclidean_dist(next_robot_pose,
                                               object_pose)
                    if next_dist < cur_dist:
                        preferences.add((motion_action,
                                         self.num_visits_init, self.val_init))
        return preferences


class AdversarialActionPrior(pomdp_py.ActionPrior):
    def __init__(self, object_id, robot_id, motion_policy,
                 num_visits_init, val_init, look_after_move=False):
        """
        Object `object_id` is trying to act adversarially against the robot `robot_id`.
        """
        self.object_id = object_id
        self.robot_id = robot_id
        self.motion_policy = motion_policy
        self.num_visits_init = num_visits_init
        self.val_init = val_init
        self._look_after_move = look_after_move

    def get_preferred_actions(self, state, history):
        # Prefer actions that move the target away from the robot.
        cur_dist = euclidean_dist(state.pose(self.object_id),
                                  state.pose(self.robot_id))
        
        if self._look_after_move:
            # No Look action; It's embedded in Move.
            preferences = set()
        else:
            # Always give preference to Look
            preferences = set({(LookAction(), self.num_visits_init, self.val_init)})

        neighbors = self.motion_policy.get_neighbors(state.pose(self.object_id),
                                                     self.motion_policy.motion_actions)
        for next_object_pose in neighbors:
            motion_action = neighbors[next_object_pose]
            next_dist = euclidean_dist(next_object_pose,
                                       state.pose(self.robot_id))
            if next_dist > cur_dist:
                preferences.add((motion_action, self.num_visits_init, self.val_init))
        return preferences
            
    
def unittest():
    from search_and_rescue.models.grid_map import GridMap
    from search_and_rescue.models.motion_policy import BasicMotionPolicy
    grid_map = GridMap(10, 10,
                       {0: (2,3), 5:(4,9)})
    motion_actions = create_motion_actions(can_stay=True)    
    bmp = BasicMotionPolicy(3, grid_map, motion_actions)
    policy_model = PolicyModel(3, bmp, look_after_move=False)
    state = JointState({3: VictimState(3,   (4,8,math.pi), True),
                        4: SearcherState(4, (4,4), (), True),
                        0: ObstacleState(0, (2,3)),
                        5: ObstacleState(5, (4,9))})
    print(policy_model.get_all_actions(state=state))
    print(policy_model.sample(state))
    
    greedy_action_prior = GreedyActionPrior(4, bmp, 10, 100, look_after_move=False)
    adversarial_action_prior = AdversarialActionPrior(3, 4, bmp, 10, 100, look_after_move=False)
    greedy_preferred = PreferredPolicyModel(greedy_action_prior,
                                            look_after_move=False)
    adversarial_preferred = PreferredPolicyModel(adversarial_action_prior,
                                                 look_after_move=False)
    print("Greedy preferred actions:")
    print(greedy_preferred.rollout(state, ()))
    print(greedy_preferred.action_prior.get_preferred_actions(state, ()))
    print("Adversarial preferred actions:")
    print(adversarial_preferred.rollout(state, ()))
    print(adversarial_preferred.action_prior.get_preferred_actions(state, ()))    


if __name__ == '__main__':
    unittest()
