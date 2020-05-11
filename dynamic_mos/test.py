from dynamic_mos.problem import *
from dynamic_mos.experiments.runner import *
import numpy as np
import random
import pickle

def create_two_room_world(room_width, room_height,
                          hallway_width, hallway_height):
    if hallway_height > room_height:
        raise ValueError("hallway height should be smaller than room's.")

    arr = np.ones((room_height,
                   room_width * 2 + hallway_width,)).astype(int)
    # Left Room
    arr[:, :room_width] = 0

    # Right Room
    arr[:, room_width + hallway_width : room_width*2 + hallway_width] = 0
    
    # Hallway
    room_mid = room_height // 2    
    if hallway_height == 1:
        arr[room_mid : room_mid + hallway_height,
            room_width : room_width + hallway_width] = 0
    else:
        hallway_mid = hallway_height // 2
        arr[room_mid - hallway_mid : room_mid,
            room_width : room_width + hallway_width] = 0
        

    lines = []
    for y in range(arr.shape[0]):
        line = []
        for x in range(arr.shape[1]):
            if arr[y,x] == 0:
                line.append(".")
            else:
                line.append("x")
        lines.append(line)

    # Create the string
    finalstr = []
    for row_chars in lines:
        finalstr.append("".join(row_chars))
    finalstr = "\n".join(finalstr)

    # get a set of free locations
    free_locations = set(tuple(reversed(loc))
                         for loc in np.vstack(np.where(arr == 0)).transpose())
    return finalstr, free_locations


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

def test_single(case, ntrials=1, planner_type="pouct", sensor_range=4):
    results = []
    for i in range(ntrials):
        mapstr, free_locations = create_two_room_world(*case)
        robot_pose = random.sample(free_locations, 1)[0]
        objD_pose = random.sample(free_locations - {robot_pose}, 1)[0]
        objE_pose = random.sample(free_locations - {robot_pose, objD_pose}, 1)[0]
        world = (place_objects(mapstr,
                               {"r": robot_pose,
                                "D": objD_pose,
                                "E": objE_pose}),
                 "r",
                 {"D": ("random", 0.4),
                  "E": ("random", 0.4)})
        _total_reward = unittest(world, planner_type=planner_type,
                                 sensor_range=sensor_range)
        results.append(_total_reward)
    return results
    

if __name__ == "__main__":
    test_single((6,6,6,2), ntrials=1, planner_type="pouct")
    # test()
    
