import pomdp_py
import copy
from ..models.transition_model import *
from ..models.dynamic_transition_model import *
from ..models.reward_model import *
from ..models.components.sensor import *
from ..models.components.motion_policy import *
from ..domain.state import *

# This is a JSON-formatted String
world0 = {
    "worldstr": [["rx..."],
                 [".x.xT"],
                 ["....."]],
    "sensors": {
        "r": "None",
    },
    "dynamic_objects": [
        {"char": "T",
         "region": [["rx@@@"],
                    [".x@xT"],
                    ["..@@@"]],
         "motion_policy": {
             "type": "deterministic",
             "path": "None",
             "actions": [],
             "params": {}
         }}
    ]
}


def interpret_motion_policy(dobj_spec):
    region = set({})
    for y, line in enumerate(dobj_spec["region"]):
        for x, c in enumerate(line):
            if c == "@" or c == dobj_spec["char"]:
                region.add((x,y))
    # possibilities:
    ## Have a path and deterministic
    ## Have a path and stochaistic
    ## No path, have action space, and deterministic
    ## No path, have action space and stochaistic
    ## No path, no action space, and deterministic
    ## No path, no action space and stochaistic
    # kinds of stochaisticity:
    ## Stops instead of move
    ## Moves randomly
    ## ... This list is very long. Hard-coding is meaningless.
    ## I don't need to enumerate this. I just need to
    ## create a few worlds that contain representative
    ## kinds of dynamic objects and see how the robot does.
    ## Would the robot be able to do something interesting?
    ## The interesting thing is that in dynamic object search
    ## it is sometimes (often) not obvious what the optimal
    ## policy should be.
    


def interpret_robot_id(robot_name):
    return -ord(robot_name)

def interpret_sensor(sensor_setting, robot_id):
    if "," in sensor_setting:
        raise ValueError("Wrong Fromat. Should not have ','. Separate tokens with space.")
    sensor_type = sensor_setting.split(" ")[0].strip()
    sensor_params = {}
    for token in sensor_setting.split(" ")[1:]:
        param_name = token.split("=")[0].strip()
        param_value = eval(token.split("=")[1].strip())
        sensor_params[param_name] = param_value

    if sensor_type == "laser":
        sensor = Laser2DSensor(robot_id, **sensor_params)
    elif sensor_type == "proximity":
        sensor = ProximitySensor(robot_id, **sensor_params)
    else:
        raise ValueError("Unknown sensor type %s" % sensor_type)
    return sensor


def interpret_dynamic_world(world_spec):
    w, l = len(worldlines[0]), len(worldlines)
    
    objects = {}    # objid -> pose
    obstacles = set({})  # objid
    robots = {}  # robot_id -> pose
    sensors = {}  # robot_id -> Sensor
    dynamic_objects = set({})  # objid -> motion_policy

    # dynamic object characters
    dynamic_object_chars = set({do["char"] for do in world_spec["dynamic_objects"]})

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
                objects[objid] = ObjectState(objid, "obstacle", (x,y))
                obstacles.add(objid)
                
            elif c.isupper():
                # target object
                objid = len(objects)
                objects[objid] = ObjectState(objid, "target", (x,y))
                if c in dynamic_object_chars:
                    print("WARNING! THIS IS A DYNAMIC OBJECT. CODE ONGOING")
                    dynamic_objects[objid] = interpret_motion_policy(world_spec["dynamic_objects"][c])
                    objects[objid]["time"] = 0
                
            elif c.islower():
                # robot
                robot_id = interpret_robot_id(c)
                robots[robot_id] = RobotState(robot_id, (x,y,0), (), None)

            else:
                assert c == ".", "Unrecognized character %s in worldstr" % c
    if len(robots) == 0:
        raise ValueError("No initial robot pose!")
    if len(objects) == 0:
        raise ValueError("No object!")

    for robot_name in world_spec["sensors"]:
        robot_id = interpret_robot_id(robot_name)
        sensors[robot_id] = interpret_sensor(world_spec["sensors"][robot_name],
                                             robot_id)
    # Make init state
    init_state = MosOOState(robot_id, {**objects, **robots})
    return MosEnvironment((w,l),
                          init_state, sensors,
                          obstacles=obstacles,
                          dynamic_objects=dynamic_objects)
