from dynamic_mos.dynamic_worlds import *
from dynamic_mos.experiments.runner import unittest
from dynamic_mos.experiments.world_types import *
import numpy as np
import random
import pickle

def test(planner_type="pouct", sensor_range=4):
    NTRIALS = 30
    cases = [(2, 2, 2, 1), (3, 3, 3, 1), (4, 4, 4, 1), (5, 5, 5, 2), (6, 6, 6, 2)]
    results = {}
    for case in cases:
        results[case] = {"trials": test_single(case, ntrials=NTRIALS,
                                               planner_type=planner_type,
                                               sensor_range=sensor_range)}
        results[case]["mean"] = np.mean(results[case]["trials"])
        results[case]["std"] = np.std(results[case]["trials"])
        results[case]["ntrials"] = np.std(NTRIALS)
    print(results)
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

def test_single(case, ntrials=1, planner_type="pouct",
                sensor_range=4, planning_time=0.7,
                discount_factor=0.99, dynamics="random",
                max_depth=40, belief_rep="histogram",
                world_type="two_rooms"):
    results = []
    for i in range(ntrials):
        if world_type == "two_rooms":
            mapstr, free_locations = create_two_room_world(*case)
        elif world_type == "loop":
            mapstr, free_locations = create_loop_world(*case)
        elif world_type == "two_rooms_loop":
            mapstr, free_locations = create_two_room_loop_world(*case)
        elif world_type == "free":
            mapstr, free_locations = create_free_world(*case)
        elif world_type == "hallway":
            mapstr, free_locations = create_hallway_world(*case)
        elif world_type == "connected_hallway":
            mapstr, free_locations = create_connected_hallway_world(*case)                        
        robot_pose = random.sample(free_locations, 1)[0]
        objD_pose = random.sample(free_locations - {robot_pose}, 1)[0]
        objE_pose = random.sample(free_locations - {robot_pose, objD_pose}, 1)[0]

        if dynamics == "goal":
            objD_goal = random.sample(free_locations - {robot_pose, objD_pose, objE_pose}, 1)[0]
            objE_goal = random.sample(free_locations - {robot_pose, objD_pose, objE_pose}, 1)[0]
            motion_spec = {"D": ("goal", (objD_goal, 0.5)),
                           "E": ("goal", (objE_goal, 0.5))}
        elif dynamics == "adversarial":
            rule = "avoid"
            motion_spec = {"D": ("adversarial", (0.2, rule)),
                           "E": ("adversarial", (0.2, rule))}
        else:
            motion_spec = {"D": ("random", 0.4),
                           "E": ("random", 0.1)}
        world = (place_objects(mapstr,
                               {"r": robot_pose,
                                "D": objD_pose,
                                "E": objE_pose}),
                 "r", motion_spec)
        _total_reward = unittest(world, planner_type=planner_type,
                                 sensor_range=sensor_range,
                                 planning_time=planning_time,
                                 belief_rep=belief_rep,
                                 discount_factor=discount_factor,
                                 look_after_move=True,
                                 max_depth=max_depth)
        results.append(_total_reward)
    return results

def test_particular_world(world, planner_type="pouct", sensor_range=4,
                          planning_time=0.7, discount_factor=0.99,
                          look_after_move=False, max_depth=20,
                          belief_rep="histogram"):
    _total_reward = unittest(world, planner_type=planner_type,
                             sensor_range=sensor_range,
                             planning_time=planning_time,
                             discount_factor=discount_factor,
                             belief_rep=belief_rep,                             
                             look_after_move=look_after_move,
                             max_depth=max_depth)
    

if __name__ == "__main__":
    random.seed(90595)
    
    # test_single((8,8,8,1), world_type="two_rooms", ntrials=1, planner_type="pouct#preferred", sensor_range=2,
    #             planning_time=0.9, discount_factor=0.95, dynamics="random",
    #             max_depth=20, belief_rep="weighted_particles")

    # test_single((5,5,5,1), world_type="two_rooms", ntrials=1, planner_type="pouct#preferred", sensor_range=2,
    #             planning_time=0.7, discount_factor=0.95, dynamics="goal",
    #             max_depth=20, belief_rep="histogram")

    # test_single((5,5,3,1,1), world_type="two_rooms_loop", ntrials=1, planner_type="greedy", sensor_range=2,
    #             planning_time=0.7, discount_factor=0.95, dynamics="adversarial",
    #             max_depth=20, belief_rep="histogram")
    
    test_single((20,20), world_type="free", ntrials=1, planner_type="pouct#preferred", sensor_range=2,
                planning_time=0.7, discount_factor=0.95, dynamics="adversarial",
                max_depth=20, belief_rep="histogram")

    # test_single((9, 1), world_type="hallway", ntrials=1, planner_type="greedy", sensor_range=2,
    #             planning_time=0.7, discount_factor=0.95, dynamics="adversarial",
    #             max_depth=20, belief_rep="histogram")

    # test_single((9, 1), world_type="connected_hallway", ntrials=1, planner_type="greedy", sensor_range=2,
    #             planning_time=0.7, discount_factor=0.95, dynamics="adversarial",
    #             max_depth=20, belief_rep="histogram")
    
    
    # test()
    # test_particular_world(dynamic_world_9,
    #                       planner_type="greedy",
    #                       sensor_range=2,
    #                       planning_time=0.7,
    #                       discount_factor=0.95,
    #                       look_after_move=True,
    #                       max_depth=20)
