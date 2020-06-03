from sciex import Trial
import random
from dynamic_mos.env.env import *
from dynamic_mos.domain.action import create_motion_actions
from dynamic_mos.models.components.motion_policy import *
from dynamic_mos.experiments.world_types import *
from adversarial_mos import *
from dynamic_mos.dynamic_worlds import *


class AdversarialTrial(Trial):

    def __init__(self, name, config, verbose=False):
        super().__init__(name, config, verbose=verbose)


    def run(self, logging=False):

        robot_char = "r"
        robot_id = interpret_robot_id(robot_char)        
        mapstr, free_locations = create_two_room_world(5,5,5,1)

        robot_pose = random.sample(free_locations, 1)[0]
        objD_pose = random.sample(free_locations - {robot_pose}, 1)[0]
        objE_pose = random.sample(free_locations - {robot_pose, objD_pose}, 1)[0]        

        # place objects
        mapstr = place_objects(mapstr,
                               {"r": robot_pose,
                                "D": objD_pose,
                                "E": objE_pose})
        
        sensorstr = make_laser_sensor(90, (1, 3), 0.5, False)
        worldstr = equip_sensors(mapstr, {robot_char: sensorstr})
        big = 100
        small = 1
        
        # interpret the world string
        dim, robots, objects, obstacles, sensors, _\
            = interpret(worldstr, {})

        # grid map
        grid_map = GridMap(dim[0], dim[1],
                           {objid: objects[objid].pose
                            for objid in obstacles})
        
        # init state
        init_state = MosOOState(robot_id, {robot_id: robots[robot_id],
                                           **objects})

        # create env
        motion_actions = create_motion_actions(scheme="xy")        
        env = AdversarialMosEnvironment(init_state,
                                        robot_id,
                                        sensors[robot_id],
                                        grid_map,
                                        motion_actions,
                                        big=big, small=small)

        # create agents
        targets = {}
        for objid in env.target_objects:
            obj_sensor = copy.deepcopy(sensors[robot_id])
            obj_sensor.robot_id = objid
            target = AdversarialTarget(objid,
                                       init_state.object_states[objid],
                                       obj_sensor,
                                       motion_actions,
                                       grid_map,
                                       robot_id,
                                       prior={robot_id: {init_state.pose(robot_id): 1.0}},
                                       action_prior=AdversarialActionPrior(objid, robot_id,
                                                                           grid_map, 10, big,
                                                                           motion_actions))
            targets[objid] = target

        robot = Searcher(robot_id,
                         init_state.object_states[robot_id],
                         sensors[robot_id],
                         grid_map,
                         env.target_objects,
                         motion_policies={objid: AdversarialPolicy(grid_map,
                                                                   sensors[robot_id].max_range,
                                                                   pr_stay=0.0,
                                                                   rule="avoid",
                                                                   motion_actions=motion_actions)
                                          for objid in objects})
        
        # solve
        
        
if __name__ == "__main__":
    trial = AdversarialTrial("a_1_b", {})
    trial.run()
