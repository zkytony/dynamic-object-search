# Dynamic object transition model
import pomdp_py
from .transition_model import StaticObjectTransitionModel, RobotTransitionModel
from .components.motion_policy import AdversarialPolicy
from ..domain.state import *

class DynamicAgentTransitionModel(pomdp_py.TransitionModel):
    """The difference between DynamicAgentTransitionModel
    and DynamicObjectTransitionModel is that the motion
    policy of the agent is conditioned on the action."""
    def __init__(self, objid, motion_policy, epsilon=1e-9):
        self._objid = objid
        self._motion_policy = motion_policy
        self._epsilon = epsilon
        
    def probability(self, next_object_state, state, action):
        return self._motion_policy.probability(next_object_state,
                                               state, action)

    def sample(self, state, action, argmax=False):
        if argmax:
            sample_func = self._motion_policy.argmax
        else:
            sample_func = self._motion_policy.random
        next_state = sample_func(state, action)
        return next_state
    

class DynamicObjectTransitionModel(pomdp_py.TransitionModel):

    """
    For now, the dynamics of an object is independent from 
    all other objects
    """

    def __init__(self, objid, motion_policy, epsilon=1e-9):
        self._objid = objid
        self._motion_policy = motion_policy
        self._epsilon = epsilon

    def probability(self, next_object_state, state, action, robot_state=None):
        if isinstance(state, pomdp_py.OOState):
            object_state = state.object_states[self._objid]
            robot_state = state.robot_state
        else:
            assert isinstance(state, ObjectState)
            object_state = state
            assert robot_state is not None

        if isinstance(self._motion_policy, AdversarialPolicy):
            return self._motion_policy.probability(next_object_state,
                                                   object_state, robot_state)
        else:
            return self._motion_policy.probability(next_object_state,
                                                   object_state)

    def sample(self, state, action, argmax=False, robot_state=None):
        if isinstance(state, pomdp_py.OOState):
            object_state = state.object_states[self._objid]
            robot_state = state.robot_state
        else:
            assert isinstance(state, ObjectState)
            object_state = state
            assert robot_state is not None

        if argmax:
            sample_func = self._motion_policy.argmax
        else:
            sample_func = self._motion_policy.random

        if isinstance(self._motion_policy, AdversarialPolicy):
            return sample_func(object_state, robot_state)
        else:
            return sample_func(object_state)

    def argmax(self, state, action):
        return self.sample(state, action, argmax=True)        


class DynamicMosTransitionModel(pomdp_py.OOTransitionModel):
    def __init__(self,
                 dim, sensors,
                 static_object_ids,
                 motion_policies,
                 epsilon=1e-9,
                 look_after_move=False):
        """
        sensors (dict): robot_id -> Sensor
        static_object_ids (set): Set of static object ids
        motion_policies (dict): Map from dynamic object id to MotionPolicy
        """
        self._sensors = sensors
        self.dynamic_object_ids = set(motion_policies.keys())
        self.dynamic_object_motion_policies = motion_policies
        transition_models = {}
        for objid in static_object_ids:
            if objid not in sensors:
                transition_models[objid] = StaticObjectTransitionModel(objid, epsilon=epsilon)
        for objid in self.dynamic_object_ids:
            if objid not in sensors:
                transition_models[objid] = DynamicObjectTransitionModel(objid,
                                                                        motion_policies[objid],
                                                                        epsilon=epsilon)        
        assert len(sensors) == 1, "Only deal with one robot in this domain!"
        robot_id = list(sensors.keys())[0]
        transition_models[robot_id] = RobotTransitionModel(sensors[robot_id],
                                                           dim,
                                                           epsilon=epsilon,
                                                           look_after_move=look_after_move)
        super().__init__(transition_models)

    def sample(self, state, action, **kwargs):
        oostate = pomdp_py.OOTransitionModel.sample(self, state, action, **kwargs)
        return MosOOState(state.robot_id, oostate.object_states)

    def argmax(self, state, action, normalized=False, **kwargs):
        oostate = pomdp_py.OOTransitionModel.argmax(self, state, action, **kwargs)
        return MosOOState(state.robot_id, oostate.object_states)

    def motion_policy(self, objid):
        return self.dynamic_object_motion_policies.get(objid, None)
