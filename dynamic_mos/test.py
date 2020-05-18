from dynamic_mos.dynamic_worlds import *
from dynamic_mos.experiments.runner import create_two_room_world, unittest
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
                max_depth=40):
    results = []
    for i in range(ntrials):
        mapstr, free_locations = create_two_room_world(*case)
        robot_pose = random.sample(free_locations, 1)[0]
        objD_pose = random.sample(free_locations - {robot_pose}, 1)[0]
        objE_pose = random.sample(free_locations - {robot_pose, objD_pose}, 1)[0]

        if dynamics == "goal":
            objD_goal = random.sample(free_locations - {robot_pose, objD_pose, objE_pose}, 1)[0]
            objE_goal = random.sample(free_locations - {robot_pose, objD_pose, objE_pose}, 1)[0]
            motion_spec = {"D": ("goal", (objD_goal, 0.2)),
                           "E": ("goal", (objE_goal, 0.2))}
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
                                 discount_factor=discount_factor,
                                 look_after_move=False,
                                 max_depth=max_depth)
        results.append(_total_reward)
    return results

def test_particular_world(world, planner_type="pouct", sensor_range=4,
                          planning_time=0.7, discount_factor=0.99,
                          look_after_move=False, max_depth=20):
    _total_reward = unittest(world, planner_type=planner_type,
                             sensor_range=sensor_range,
                             planning_time=planning_time,
                             discount_factor=discount_factor,
                             look_after_move=look_after_move,
                             max_depth=max_depth)
    

if __name__ == "__main__":
    random.seed(90595)
    test_single((8,8,8,1), ntrials=1, planner_type="pouct#preferred",
                planning_time=0.9, discount_factor=0.95, dynamics="goal")
    # test()
    # test_particular_world(dynamic_world_8,
    #                       planner_type="pouct#preferred",
    #                       sensor_range=2,
    #                       planning_time=0.7,
    #                       discount_factor=0.95,
    #                       look_after_move=True,
    #                       max_depth=20)
