import pomdp_py
from search_and_rescue.env.action import *
from search_and_rescue.env.state import *
from search_and_rescue.models.motion_policy import BasicMotionPolicy
from search_and_rescue.models.transition_model import *
from search_and_rescue.models.sensor import *
from search_and_rescue.models.reward_model import *

class MultiAgentRewardModel(pomdp_py.RewardModel):
    def __init__(self, reward_models):
        self._reward_models = reward_models

    def __contains__(self, objid):
        return objid in self._reward_models

    def sample(self, state, action, next_state, agent_id=None):
        return self._reward_models[agent_id].sample(state, action, next_state)

class SAREnvironment(pomdp_py.Environment):
    def __init__(self,
                 role_to_ids,
                 grid_map,
                 sensors,
                 init_state,
                 transition_model,
                 reward_model):
        self._role_to_ids = role_to_ids
        self.grid_map = grid_map
        self.sensors = sensors
        super().__init__(init_state,
                         transition_model,
                         reward_model)

    def state_transition(self, action, execute=True):
        assert isinstance(action, ActionCollection)
        
        next_state = copy.deepcopy(self.state)
        active_agents = set(next_state.active_agents)
        # First, check which agents are still active;
        # FOR RIGHT NOW:
        # A searcher never becomes inactive;
        # A suspect becomes inactive when it falls into the objects_found set of the searcher
        # A victim becomes inactive when it falls into the fov_objects set of the suspect
        rewards = {}
        for agent_id in action.actions:
            next_object_state = self.transition_model[agent_id].sample(self.state, action[agent_id])                            
            if isinstance(action[agent_id], FindAction):
                for found_agent_id in next_object_state["objects_found"]:
                    active_agents.discard(found_agent_id)
            else:
                if self.role_for(agent_id) == "suspect":
                    for fov_agent_id in next_object_state["fov_objects"]:
                        if fov_agent_id in self._role_to_ids["victim"]:
                            active_agents.discard(fov_agent_id)
            next_state.set_object_state(agent_id, next_object_state)
            if agent_id in self.reward_model:
                rewards[agent_id] = self.reward_model.sample(
                    self.state, action[agent_id], next_state,
                    agent_id=agent_id)
        next_state.active_agents = active_agents
        if execute:
            self.apply_transition(next_state)
            return rewards

        else:
            return next_state, rewards

    def ids_for(self, role):
        return self._role_to_ids[role]

    def role_for(self, objid):
        return self.state.object_states[objid].objclass
        
    @classmethod
    def construct(self, role_to_ids, init_object_states,
                  grid_map, motion_actions, sensors,
                  look_after_move=False):
        assert len(role_to_ids["searcher"]) <= 1, "Currently, only one searcher is allowed."
        if "target" not in role_to_ids:
            role_to_ids["target"] = set()
            
        static_object_ids = set(grid_map.obstacles.keys()) | role_to_ids["target"]
        dynamic_object_ids = role_to_ids["searcher"] | role_to_ids["victim"] | role_to_ids["suspect"]

        # The transition models of the environment; OOTransitionmodel, but not agent-specific
        transition_models = {}
        reward_models = {}
        for objid in init_object_states:
            if objid in static_object_ids:
                t = StaticTransitionModel(objid)
            else:
                motion_policy = BasicMotionPolicy(objid, grid_map, motion_actions)
                t = DynamicAgentTransitionModel(objid, motion_policy, sensors[objid],
                                                look_after_move=look_after_move)
            transition_models[objid] = t

            if objid in role_to_ids["searcher"]:
                r = SearcherRewardModel(objid, role_to_ids=role_to_ids)
            if objid in role_to_ids["victim"]:
                r = VictimRewardModel(objid, role_to_ids=role_to_ids)            
            if objid in role_to_ids["suspect"]:
                r = SuspectRewardModel(objid, role_to_ids=role_to_ids)                
            reward_models[objid] = r
        tmodel = pomdp_py.OOTransitionModel(transition_models)
        rmodel = MultiAgentRewardModel(reward_models)

        init_state = EnvState(
            tuple(role_to_ids["searcher"] | role_to_ids["victim"] | role_to_ids["suspect"]),
            JointState(init_object_states)
        )
        return SAREnvironment(role_to_ids, grid_map, sensors, init_state, tmodel, rmodel)


#### Interpret string as an initial world state ####
def interpret(worldstr):
    """
    Interprets a problem instance description in `worldstr`
    and returns the corresponding MosEnvironment.

    For example: This string
    
    .. code-block:: text

        Rx...
        .x.xV
        .S...
        ***
        R: laser fov=90 min_range=1 max_range=10
        V: laser fov=90 min_range=1 max_range=10
        S: laser fov=90 min_range=1 max_range=10
    
    describes a 3 by 5 world where x indicates obsticles and R,S,V indicate
    searcher, victim, and suspect. 

    After the world, the :code:`***` signals description of the sensor for each robot.
    For example "R laser 90 1 10" means that robot `R` will have a Laser2Dsensor
    with fov 90, min_range 1.0, and max_range of 10.0.    

    Args:
        worldstr (str): a string that describes the initial state of the world.

    Returns:
        (w,l), robots, objects, obstacles, sensors, motion_policies
            
    """
    worldlines = []
    sensorlines = []
    mode = "world"
    for line in worldstr.splitlines():
        line = line.strip()
        if len(line) > 0:
            if line == "***":
                mode = "sensor"
                continue
            if mode == "world":
                worldlines.append(line)
            if mode == "sensor":
                sensorlines.append(line)
    
    lines = [line for line in worldlines
             if len(line) > 0]
    w, l = len(worldlines[0]), len(worldlines)
    
    objects = {}    # objid -> ObjectState(pose)
    obstacles = set({})  # objid
    robots = {}  # robot_id -> RobotState(pose)
    char_to_robots = {}  # str -> set of robot ids
    sensors = {}  # robot_id -> Sensor
    role_to_ids = {"suspect":set(),
                   "searcher":set(),
                   "victim":set(),
                   "target":set()}  # role (str) -> set of agent ids

    # Parse world
    for y, line in enumerate(worldlines):
        if len(line) != w:
            raise ValueError("World size inconsistent."\
                             "Expected width: %d; Actual Width: %d"
                             % (w, len(line)))
        for x, c in enumerate(line):
            if c == "x":
                # obstacle
                objid = 1000 + len(obstacles)  # obstacle id
                objects[objid] = ObstacleState(objid, (x,y))
                obstacles.add(objid)

            elif c.isupper():
                if c == "R":
                    # searcher
                    objid = len(role_to_ids["searcher"]) + 7000
                    robots[objid] = SearcherState(objid, (x,y,0), (), False)
                    role_to_ids["searcher"].add(objid)
                    
                elif c == "V":
                    # victim
                    objid = len(role_to_ids["victim"]) + 3000
                    robots[objid] = VictimState(objid, (x,y,0), (), False)
                    role_to_ids["victim"].add(objid)                    
                    
                elif c == "S":
                    # suspect
                    objid = len(role_to_ids["suspect"]) + 5000
                    robots[objid] = SuspectState(objid, (x,y,0), (), False)
                    role_to_ids["suspect"].add(objid)
                    
                else:
                    objid = len(role_to_ids["target"])
                    objects[objid] = TargetState(objid, (x,y))
                    role_to_ids["target"].add(objid)

                if c in {"R", "V", "S"}:
                    if c not in char_to_robots:
                        char_to_robots[c] = set()
                    char_to_robots[c].add(objid)
                
            else:
                assert c == ".", "Unrecognized character %s in worldstr" % c
    if len(robots) == 0:
        raise ValueError("No initial robot pose!")

    # Parse sensors
    for line in sensorlines:
        if "," in line:
            raise ValueError("Wrong Fromat. SHould not have ','. Separate tokens with space.")
        robot_char = line.split(":")[0].strip()
        sensor_setting = line.split(":")[1].strip()
        sensor_type = sensor_setting.split(" ")[0].strip()
        sensor_params = {}
        for token in sensor_setting.split(" ")[1:]:
            param_name = token.split("=")[0].strip()
            param_value = eval(token.split("=")[1].strip())
            sensor_params[param_name] = param_value

        for robot_id in char_to_robots[robot_char]:
            if sensor_type == "laser":
                sensor = Laser2DSensor(robot_id, **sensor_params)
            elif sensor_type == "proximity":
                sensor = ProximitySensor(robot_id, **sensor_params)
            else:
                raise ValueError("Unknown sensor type %s" % sensor_type)
            sensors[robot_id] = sensor
    return (w,l), robots, objects, obstacles, sensors, role_to_ids


#### Utility functions for building the worldstr ####
def equip_sensors(worldmap, sensors):
    """
    Args:
        worldmap (str): a string that describes the initial state of the world.
        sensors (dict) a map from robot character representation (e.g. 'r') to a
    string that describes its sensor (e.g. 'laser fov=90 min_range=1 max_range=5
    angle_increment=5')

    Returns:
        str: A string that can be used as input to the `interpret` function
    """
    worldmap += "\n***\n"
    for robot_char in sensors:
        worldmap += "%s: %s\n" % (robot_char, sensors[robot_char])
    return worldmap

def make_laser_sensor(fov, dist_range, angle_increment, occlusion):
    """
    Returns string representation of the laser scanner configuration.
    For example:  "laser fov=90 min_range=1 max_range=10"

    Args:
        fov (int or float): angle between the start and end beams of one scan (degree).
        dist_range (tuple): (min_range, max_range)
        angle_increment (int or float): angular distance between measurements (rad).
        occlusion (bool): True if consider occlusion

    Returns:
        str: String representation of the laser scanner configuration.
    """
    fovstr = "fov=%s" % str(fov)
    rangestr = "min_range=%s max_range=%s" % (str(dist_range[0]), str(dist_range[1]))
    angicstr = "angle_increment=%s" % (str(angle_increment))
    occstr = "occlusion_enabled=%s" % str(occlusion)
    return "laser %s %s %s %s" % (fovstr, rangestr, angicstr, occstr)

def make_proximity_sensor(radius, occlusion):
    """
    Returns string representation of the proximity sensor configuration.
    For example: "proximity radius=5 occlusion_enabled=False"

    Args:
        radius (int or float)
        occlusion (bool): True if consider occlusion
    Returns:
        str: String representation of the proximity sensor configuration.
    """
    radiustr = "radius=%s" % str(radius)
    occstr = "occlusion_enabled=%s" % str(occlusion)
    return "proximity %s %s" % (radiustr, occstr)


####### UNIT TESTS ######
worldstr=\
"""
Rx...
.x.xV
.S...    
"""
def unittest(worldstr):
    from search_and_rescue.models.grid_map import GridMap
    from search_and_rescue.env.action import create_motion_actions

    laserstr = make_laser_sensor(90, (1, 50), 0.5, False)
    worldstr = equip_sensors(worldstr, {"S": laserstr,
                                        "V": laserstr,
                                        "R": laserstr})
    dims, robots, objects, obstacles, sensors, role_to_ids = interpret(worldstr)
    grid_map = GridMap(dims[0], dims[1],
                       {objid: objects[objid].pose
                        for objid in obstacles})
    motion_actions = create_motion_actions(can_stay=True)
    env = SAREnvironment.construct(role_to_ids,
                                   {**robots, **objects},
                                   grid_map, motion_actions, sensors,
                                   look_after_move=True)
    print(env.state)
    

if __name__ == '__main__':
    unittest(worldstr)
