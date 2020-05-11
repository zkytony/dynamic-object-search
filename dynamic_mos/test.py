from dynamic_mos.problem import *
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
    print(finalstr)

    # get a set of free locations
    free_locations = set(tuple(reversed(loc))
                         for loc in np.vstack(np.where(arr == 0)).transpose())
    return finalstr, free_locations


def test():
    NTRIALS = 30
    cases = {(2, 2, 2, 1), (3, 3, 3, 1), (4, 4, 4, 1), (5, 5, 5, 2), (6, 6, 6, 2)}
    results = {}
    for args in cases:
        results[args] = {"trials": []}
        for i in range(NTRIALS):
            mapstr, free_locations = create_two_room_world(*args)
            robot_pose = random.sample(free_locations, 1)[0]
            world = (place_objects(mapstr,
                                   {"r": random.sample(free_locations, 1)[0],
                                    "D": random.sample(free_locations, 1)[0],
                                    "E": random.sample(free_locations, 1)[0]}),
                     "r",
                     {"D": ("random", 0.4),
                      "E": ("random", 0.4)})
            _total_reward = unittest(world)
            results[args]["trials"].append(_total_reward)
        results[args]["mean"] = np.mean(results[args]["trials"])
        results[args]["std"] = np.std(results[args]["trials"])
        results[args]["ntrials"] = np.std(NTRIALS)
    print(results)
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    test()
    
