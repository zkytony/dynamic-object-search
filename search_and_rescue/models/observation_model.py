import pomdp_py
import random
import math
import numpy as np
from search_and_rescue.env.observation import *
from search_and_rescue.env.state import *
from search_and_rescue.env.action import *


def is_sensing(look_after_move, action):
    return isinstance(action, LookAction) or isinstance(action, FindAction)\
        or (look_after_move and isinstance(action, MotionAction))

class SensorModel(pomdp_py.OOObservationModel):
    """For a given sensor, construct an observation model"""
    def __init__(self, object_ids, sensor, grid_map, look_after_move=False):
        self._sensor = sensor
        self.agent_id = sensor.agent_id
        self._grid_map = grid_map        
        self._look_after_move = look_after_move
        observation_models = {}
        for objid in object_ids:
            observation_models[objid] = ObjectSensorModel(objid, sensor, grid_map,
                                                         look_after_move=look_after_move)
        pomdp_py.OOObservationModel.__init__(self, observation_models)            

    def sample(self, next_state, action, argmax=False):
        if not is_sensing(self._look_after_move, action):
            return JointObservation({})

        factored_observations = super().sample(next_state, action, argmax=argmax)
        return JointObservation.merge(factored_observations, next_state)

class ObjectSensorModel(pomdp_py.ObservationModel):
    """Object sensor model; It samples the observation for a particular object `objid`"""
    def __init__(self, objid, sensor, grid_map, look_after_move=False):
        self._objid = objid
        self._sensor = sensor  # Sensor belongs to the agent.
        self._grid_map = grid_map
        self._look_after_move = look_after_move

    def probability(self, obj_observation, next_state, action, **kwargs):
        assert isinstance(obj_observation, ObjectObservation)
        if obj_observation.objid != self._objid:
            import pdb; pdb.set_trace()
        if not is_sensing(self._look_after_move, action):
            # No observation should be received
            if obj_observation.pose == ObjectObservation.NULL:
                return 1.0
            else:
                return 0.0
        
        return self._sensor.probability(obj_observation,
                                        next_state,
                                        action, **kwargs)
    
    def sample(self, next_state, action, argmax=False):
        if not is_sensing(self._look_after_move, action):
            # Not a look action. So no observation
            return ObjectObservation(self._objid, ObjectObservation.NULL)
        
        if argmax:
            sample_func = self._sensor.mpe
        else:
            sample_func = self._sensor.random
        return sample_func(next_state, action, object_id=self._objid,
                           grid_map=self._grid_map)

    def argmax(self, next_state, action):
        return self.sample(next_state, action, argmax=True)

def unittest():
    from search_and_rescue.models.sensor import Laser2DSensor
    from search_and_rescue.models.grid_map import GridMap
    grid_map = GridMap(10, 10,
                       {0: (2,3), 5:(4,9)})
    motion_actions = create_motion_actions(can_stay=True)

    sensor = Laser2DSensor(3, fov=359, max_range=10, epsilon=0.8, sigma=0.2)
    o = SensorModel({3,4,0,5}, sensor, grid_map)
    next_state = JointState({3: VictimState(3, (4,8,math.pi), (), True),
                             4: SearcherState(4, (4,4), (), True),
                             0: ObstacleState(0, (2,3)),
                             5: ObstacleState(5, (4,9))})
    action = Look
    print(o.sample(next_state, action))
    print(o.argmax(next_state, action))


if __name__ == '__main__':
    unittest()
