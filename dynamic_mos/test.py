from dynamic_mos.problem import *
import numpy as np
import random

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
    mapstr, free_locations = create_two_room_world(5, 5, 4, 1)
    robot_pose = random.sample(free_locations, 1)[0]
    world = (place_objects(mapstr,
                           {"r": random.sample(free_locations, 1)[0],
                            "D": random.sample(free_locations, 1)[0],
                            "E": random.sample(free_locations, 1)[0]}),
             "r",
             {"D": ("random", 0.5),
              "E": ("random", 0.2)})
    unittest(world)

if __name__ == "__main__":
    test()
    
