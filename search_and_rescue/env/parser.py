
#### Interpret string as an initial world state ####
def interpret(worldstr, motion_policies_dict={}):
    """
    Interprets a problem instance description in `worldstr`
    and returns the corresponding MosEnvironment.

    For example: This string
    
    .. code-block:: text

        rx...
        .x.xT
        .....
        ***
        r: laser fov=90 min_range=1 max_range=10
    
    describes a 3 by 5 world where x indicates obsticles and T indicates
    the "target object". T could be replaced by any upper-case letter A-Z
    which will serve as the object's id. Lower-case letters a-z (except for x)
    serve as id for robot(s).

    After the world, the :code:`***` signals description of the sensor for each robot.
    For example "r laser 90 1 10" means that robot `r` will have a Laser2Dsensor
    with fov 90, min_range 1.0, and max_range of 10.0.    

    Args:
        worldstr (str): a string that describes the initial state of the world.

    Returns:
        MosEnvironment: the corresponding environment for the world description.
            
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
    sensors = {}  # robot_id -> Sensor
    motion_policies = {}  # objid -> tuple ("policy_type", *rgs)

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
                print("Object %s is assigned id %d" % (c, objid))
                objects[objid] = ObjectState(objid, "target", (x,y))
                if c in motion_policies_dict:
                    if type(motion_policies_dict[c]) == list:
                        motion_policies[objid] = ("iterative", motion_policies_dict[c])
                    elif type(motion_policies_dict[c]) == tuple:
                        motion_policies[objid] = motion_policies_dict[c]
                    else:
                        raise ValueError("Invalid specification of motion policy for %s." % c)
                    objects[objid]["time"] = 0
                    print(str(motion_policies_dict[c]))
                
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

    # Parse sensors
    for line in sensorlines:
        if "," in line:
            raise ValueError("Wrong Fromat. SHould not have ','. Separate tokens with space.")
        robot_name = line.split(":")[0].strip()
        robot_id = interpret_robot_id(robot_name)
        assert robot_id in robots, "Sensor specified for unknown robot %s" % (robot_name)
        
        sensor_setting = line.split(":")[1].strip()
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
        sensors[robot_id] = sensor
    return (w,l), robots, objects, obstacles, sensors, motion_policies        

