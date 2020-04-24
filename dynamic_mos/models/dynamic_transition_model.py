# Dynamic object transition model
import pomdp_py
from .transition_model import StaticObjectTransitionModel, RobotTransitionModel
from ..domain.state import *

class DynamicObjectTransitionModel(pomdp_py.TransitionModel):

    """
    For now, the dynamics of an object is independent from 
    all other objects
    """

    def __init__(self, objid, motion_policy, epsilon=1e-9):
        self._objid = objid
        self._motion_policy = motion_policy
        self._epsilon = epsilon

    def probability(self, next_object_state, state, action):
        return self._motion_policy.probability(next_object_state,
                                               state.object_states[self._objid])

    def sample(self, state, action, argmax=False):
        cur_object_state = state.object_states[self._objid]
        if argmax:
            next_object_state = self._motion_policy.argmax(cur_object_state)
        else:
            next_object_state = self._motion_policy.random(cur_object_state)
        return next_object_state

    def argmax(self, state, action):
        return self.sample(state, action, argmax=True)        


class DynamicMosTransitionModel(pomdp_py.OOTransitionModel):
    def __init__(self,
                 dim, sensors,
                 static_object_ids,
                 dynamic_object_ids,
                 motion_policies,
                 epsilon=1e-9):
        """
        sensors (dict): robot_id -> Sensor
        for_env (bool): True if this is a robot transition model used by the
             Environment.  see RobotTransitionModel for details.
        """
        self._sensors = sensors
        transition_models = {}
        for objid in static_object_ids:
            if objid not in sensors:
                transition_models[objid] = StaticObjectTransitionModel(objid, epsilon=epsilon)
        for objid in dynamic_object_ids:
            if objid not in sensors:
                transition_models[objid] = DynamicObjectTransitionModel(objid,
                                                                        motion_policies[objid],
                                                                        epsilon=epsilon)        
        for robot_id in sensors:
            transition_models[robot_id] = RobotTransitionModel(sensors[robot_id],
                                                               dim,
                                                               epsilon=epsilon)
        super().__init__(transition_models)

    def sample(self, state, action, **kwargs):
        oostate = pomdp_py.OOTransitionModel.sample(self, state, action, **kwargs)
        return MosOOState(oostate.object_states)

    def argmax(self, state, action, normalized=False, **kwargs):
        oostate = pomdp_py.OOTransitionModel.argmax(self, state, action, **kwargs)
        return MosOOState(oostate.object_states)

