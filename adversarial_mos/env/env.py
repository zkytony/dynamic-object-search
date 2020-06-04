import pomdp_py
from adversarial_mos.adversary.agent import *
from dynamic_mos.models.transition_model import *
from dynamic_mos.models.reward_model import *

class CompositeAction(pomdp_py.Action):
    def __init__(self, actions):
        self.actions = actions

    def __getitem__(self, objid):
        if objid in self.actions:
            return self.actions[objid]
        else:
            return None

    def __str__(self):
        res = "CompositeAction\n"
        for objid in self.actions:
            res += "  %d: %s\n" % (objid, str(self.actions[objid]))
        return res

    def __repr__(self):
        return "CompositeAction(%s)" % str(self.actions)
    

class CompositeObservation(pomdp_py.Observation):
    def __init__(self, observations):
        self.observations = observations

    def __getitem__(self, objid):
        if objid in self.observations:
            return self.observations[objid]
        else:
            return None

    def __str__(self):
        res = "CompositeObservation\n"
        for objid in self.observations:
            res += "  %d: %s\n" % (objid, str(self.observations[objid]))
        return res

    def __repr__(self):
        return "CompositeObservation(%s)" % str(self.observations)
        

def compute_target_objects(grid_map, init_state):
    obstacles = set(grid_map.obstacles.keys())
    target_objects = \
        {objid
         for objid in set(init_state.object_states.keys()) - obstacles
         if not isinstance(init_state.object_states[objid], RobotState)}
    return target_objects


class AdversarialMosEnvironment(pomdp_py.Environment):

    def __init__(self, init_state,
                 robot_id, target_objects,
                 sensor, grid_map, motion_policies,
                 look_after_move=True, big=100, small=1):

        self.grid_map = grid_map
        self._robot_id = robot_id
        self.target_objects = target_objects
        # REFACTOR
        self.dynamic_object_ids = self.target_objects
        obstacles = set(grid_map.obstacles.keys())

        tmodels = {}
        for objid in {*target_objects, robot_id}:
            if objid == robot_id:
                t = RobotTransitionModel(sensor,
                                         (grid_map.width, grid_map.length),
                                         look_after_move=look_after_move)
            else:
                t = DynamicAgentTransitionModel(objid, motion_policies[objid])
            tmodels[objid] = t
        transition_model = pomdp_py.OOTransitionModel(tmodels)            

        reward_model = GoalRewardModel(self.target_objects,
                                            big=big, small=small)
        super().__init__(init_state,
                         transition_model,
                         reward_model)

        # For visualization
        self.width, self.length = grid_map.width, grid_map.length

    @property
    def robot_ids(self):
        # For visualization ... REFACTOR
        return set({self._robot_id})
                 
    def state_transition(self, action, execute=True):
        next_state = copy.deepcopy(self.state)
        for objid in self.state.object_states:
            if objid in self.transition_model.transition_models\
               and action[objid] is not None:
                if objid in self.state.robot_state["objects_found"]:
                    continue
                next_object_state = self.transition_model[objid].sample(self.state, action[objid])
                next_state.set_object_state(objid, next_object_state)

        reward = self.reward_model.sample(self.state, action[self._robot_id], next_state,
                                          robot_id=self._robot_id)
        if execute:
            self.apply_transition(next_state)
            return reward

        else:
            return next_state, reward
