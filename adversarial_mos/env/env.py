import pomdp_py
from adversarial_mos.adversary.agent import *
from dynamic_mos.models.transition_model import *

class CompositeAction(pomdp_py.Action):
    def __init__(self, actions):
        self.actions = actions

    def __getitem__(self, objid):
        if objid in self.actions:
            return self.actions[objid]
        else:
            return None

class AdversarialMosEnvironment(pomdp_py.Environment):

    def __init__(self, init_state,
                 robot_id,
                 sensor, grid_map, motion_actions,
                 look_after_move=True, big=100, small=1):

        self._robot_id = robot_id
        obstacles = set({}) if self.grid_map is None else set(grid_map.obstacles.keys())
        self.target_objects = \
            {objid
             for objid in set(init_state.object_states.keys()) - obstacles
             if not isinstance(init_state.object_states[objid], RobotState)}            
        
        tmodels = {}
        for objid in init_state.object_states:
            if objid == robot_id:
                t = MosTransitionModel((grid_map.width, grid_map.length),
                                       {robot_id: sensor},
                                       obstacles,
                                       look_after_move=look_after_move)
            else:
                t = AdversarialTransitionModel(objid,
                                               robot_id,
                                               grid_map,
                                               motion_actions)
            tmodels[objid] = t
        transition_model = pomdp_py.OOTransitionModel(tmodels)            

        reward_model = GoalRewardModel(self.target_objects,
                                            big=big, small=small)
        super().__init__(init_state,
                         transition_model,
                         reward_model)
        
                 
    def state_transition(self, action, execute=True):
        next_state = copy.deepcopy(self.state)
        for objid in self.state.object_ids:
            if objid in self.transition_model.transition_models\
               and action[objid] is not None:
                next_object_state = self.transition_model[objid].sample(self.state, action[objid])
                next_state.set_object_state(objid, next_object_state)

        reward = self.reward_model.sample(self.state, action[self._robot_id], next_state,
                                          robot_id=self._robot_id)
        if execute:
            self.apply_transition(next_state)
            return reward

        else:
            return next_state, reward
