import pomdp_py
from dynamic_mos.models.transition_model import *
from dynamic_mos.agent.agent import *

class Searcher(MosAgent):
    def __init__(self,
                 robot_id,
                 init_robot_state,
                 sensor,
                 grid_map,
                 object_ids,
                 motion_policies={},
                 **kwargs):
        self.robot_id = robot_id
        self._object_ids = object_ids
        self.sensor = sensor
        super().__init__(robot_id,
                         init_robot_state,
                         object_ids,
                         (grid_map.width, grid_map.length),
                         sensor,
                         sigma=kwargs.get("sigma", 0.01),
                         epsilon=kwargs.get("epsilon", 1),
                         prior=kwargs.get("prior", {}),
                         motion_policies=motion_policies,
                         num_particles=kwargs.get("num_particles", 1000),
                         grid_map=grid_map,
                         big=kwargs.get("big", 100),
                         small=kwargs.get("small", 1),
                         action_prior=kwargs.get("action_prior", None),
                         look_after_move=kwargs.get("look_after_move", False))
                         
        
        

