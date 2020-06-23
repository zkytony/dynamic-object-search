import pomdp_py
import copy
from search_and_rescue.env.action import *
from search_and_rescue.env.state import *

class StaticTransitionModel(pomdp_py.TransitionModel):
    """This model assumes the object is static."""
    def __init__(self, objid, epsilon=1e-9):
        self._objid = objid
        self._epsilon = epsilon

    def probability(self, next_object_state, state, action, **kwargs):
        if isinstance(state, JointState):
            object_state = state.object_states[next_object_state['id']]
        else:
            object_state = state
        if next_object_state != object_state:
            return self._epsilon
        else:
            return 1.0 - self._epsilon
    
    def sample(self, state, action):
        """Returns next_object_state"""
        return self.argmax(state, action)
    
    def argmax(self, state, action):
        """Returns the most likely next object_state"""
        return copy.deepcopy(state.object_states[self._objid])

class DynamicAgentTransitionModel(pomdp_py.TransitionModel):
    def __init__(self, agent_id, motion_policy, sensor, epsilon=1e-9, look_after_move=False):
        """
        The agent is taking the actions.
        agent_id (int) The id of the agent that this dynamics model is describing.
        """
        self._agent_id = agent_id
        self._motion_policy = motion_policy
        self._epsilon = epsilon
        self._look_after_move = look_after_move
        self._sensor = sensor
        
    def probability(self, next_object_state, state, action, **kwargs):
        return self._motion_policy.probability(next_object_state,
                                               state, action, **kwargsx)

    def sample(self, state, action, argmax=False):
        if isinstance(action, MotionAction):
            if argmax:
                sample_func = self._motion_policy.argmax
            else:
                sample_func = self._motion_policy.random
                
            next_agent_state = sample_func(state, action)
        else:
            next_agent_state = copy.deepcopy(state.object_states[self._agent_id])

        if self._look_after_move\
           or (isinstance(action, LookAction) or isinstance(action, FindAction)):
            # Camera is turned on; The robot is looking
            next_agent_state["camera_on"] = True
            
            detectables = self._detectables(next_agent_state.pose, state, self._sensor)
            if hasattr(next_agent_state, "fov_objects"):
                # The object state is not searcher.
                # Updating the objects in the field of view because the robot is looking.
                next_agent_state["fov_objects"] = tuple(detectables)
            else:
                # The object state is a searcher, not victim/suspect.
                if isinstance(action, FindAction):
                    next_agent_state["objects_found"] =\
                    tuple(set(next_agent_state["objects_found"]) | detectables)
        else:
            # Camera is turned off
            next_agent_state["camera_on"] = False
        next_agent_state["time"] = state.object_states[self._agent_id]["time"] + 1
        return next_agent_state

    def _detectables(self, agent_pose, state, sensor):
        result = set({})
        for objid in state.object_states:
            if objid == self._agent_id:
                continue
            if sensor.within_range(agent_pose, state.pose(objid)):
                result.add(objid)
        return result

class DynamicObjectTransitionModel(pomdp_py.TransitionModel):

    """
    For now, the dynamics of an object is independent from 
    all other objects
    """

    def __init__(self, objid, motion_policy, epsilon=1e-9):
        self._objid = objid
        self._motion_policy = motion_policy
        self._epsilon = epsilon

    @property
    def motion_policy(self):
        return self._motion_policy

    def probability(self, next_object_state, state, *args, **kwargs):
        return self._motion_policy.probability(next_object_state, state, **kwargs)

    def sample(self, state, *args, argmax=False):
        if argmax:
            sample_func = self._motion_policy.argmax
        else:
            sample_func = self._motion_policy.random
        next_object_state = sample_func(state)
        next_object_state["time"] = state.object_states[self._objid]["time"] + 1
        return next_object_state

    def argmax(self, state, *args):
        return self.sample(state, argmax=True)        

    
class JointTransitionModel(pomdp_py.OOTransitionModel):
    def __init__(self,
                 agent_id,
                 sensor,
                 static_object_ids,
                 dynamic_object_ids,
                 motion_policies,
                 look_after_move=False):
        self.object_ids = {"static": static_object_ids,
                           "dynamic": dynamic_object_ids}
        transition_models = {
            agent_id: DynamicAgentTransitionModel(agent_id,
                                                  motion_policies[agent_id],
                                                  sensor,
                                                  look_after_move=look_after_move)}
        for objid in static_object_ids:
            transition_models[objid] = StaticTransitionModel(objid)
        for objid in dynamic_object_ids:
            if objid != agent_id:
                transition_models[objid] = DynamicObjectTransitionModel(objid, motion_policies[objid])
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
    from search_and_rescue.models.sensor import Laser2DSensor
    from search_and_rescue.models.observation_model import SensorModel
    grid_map = GridMap(10, 10,
                       {0: (2,3), 5:(4,9)})
    motion_actions = create_motion_actions(can_stay=True)
    sensor = Laser2DSensor(4, fov=359, max_range=10, epsilon=0.8, sigma=0.2)
    t3 = JointTransitionModel(3, sensor,
                             {0,5}, {3,4,6},
                             {3: BasicMotionPolicy(3, grid_map, motion_actions),
                              4: AdversarialPolicy(4, 6, grid_map, 3, motion_actions=motion_actions, rule="chase"),
                              6: AdversarialPolicy(6, 3, grid_map, 3, motion_actions=motion_actions)})
    
    t4 = JointTransitionModel(4, sensor,
                              {0,5}, {3,4,6},
                              {4: BasicMotionPolicy(4, grid_map, motion_actions),
                               3: AdversarialPolicy(3, 6, grid_map, 3, motion_actions=motion_actions),
                               6: AdversarialPolicy(6, 4, grid_map, 3, motion_actions=motion_actions)})

    t6 = JointTransitionModel(6, sensor,
                              {0,5}, {3,4,6},
                              {6: BasicMotionPolicy(6, grid_map, motion_actions),
                               3: AdversarialPolicy(3, 4, grid_map, 3, motion_actions=motion_actions),
                               4: AdversarialPolicy(4, 6, grid_map, 3, motion_actions=motion_actions, rule="chase")})
    
    state = JointState({3: VictimState(3, (1,0,0), (), False),
                        4: SearcherState(4, (4,4,math.pi/2), (), True),
                        6: SuspectState(6, (4,8,math.pi*2/3), (), True),
                        0: ObstacleState(0, (2,3)),
                        5: ObstacleState(5, (4,9))})
    print("state:")
    print(state)
    print("OBJ 3 Victim moves from state")
    print(t3.sample(state, MoveEast))
    print("OBJ 4 Searcher moves from state")
    print(t4.sample(state, MoveEast))
    print("OBJ 6 Suspect moves from state")
    print(t6.sample(state, MoveEast))

    sp4 = t4.sample(state, MoveEast)
    assert isinstance(sp4.object_states[4], SearcherState)
    assert sp4.object_states[4]["camera_on"] is False

    # Searcher will Look and Find.
    o = SensorModel({3,4,6,0,5}, sensor, grid_map)
    print("\nOBJ 4 Searcher Look and Find")
    print("  Observation: ", o.sample(state, Look))
    print("  Look transition: ", t4.sample(state, Look).object_states[4])
    print("  Find transition: ", t4.sample(state, Find).object_states[4])    

    # Victim will Look
    sensor = Laser2DSensor(3, fov=359, max_range=10, epsilon=0.8, sigma=0.2)    
    o = SensorModel({3,4,6,0,5}, sensor, grid_map)
    print("\nOBJ 3 Victim Look")
    print("  Observation: ", o.sample(state, Look))
    print("  Look transition: ", t3.sample(state, Look).object_states[3])

    # Suspect will Look
    sensor = Laser2DSensor(6, fov=359, max_range=10, epsilon=0.8, sigma=0.2)
    o = SensorModel({3,4,6,0,5}, sensor, grid_map)    
    print("\nOBJ 6 Suspect Look")
    print("  Observation: ", o.sample(state, Look))
    print("  Look transition: ", t6.sample(state, Look).object_states[6])

if __name__ == '__main__':
    unittest()
    
