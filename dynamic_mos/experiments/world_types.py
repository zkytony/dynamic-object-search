import numpy as np
import random

def arr_to_worldstr(arr):
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


def create_two_room_world(room_width, room_length,
                          hallway_width, hallway_length):
    """Creates a world with two rooms connected by a hallway"""
    if hallway_length > room_length:
        raise ValueError("hallway length should be smaller than room's.")

    arr = np.ones((room_length,
                   room_width * 2 + hallway_width,)).astype(int)
    # Left Room
    arr[:, :room_width] = 0

    # Right Room
    arr[:, room_width + hallway_width : room_width*2 + hallway_width] = 0
    
    # Hallway
    room_mid = room_length // 2    
    if hallway_length == 1:
        arr[room_mid : room_mid + hallway_length,
            room_width : room_width + hallway_width] = 0
    else:
        hallway_mid = hallway_length // 2
        arr[room_mid - hallway_mid : room_mid + hallway_mid,
            room_width : room_width + hallway_width] = 0
    return arr_to_worldstr(arr)


def create_loop_world(width, length, loop_width):
    if width - loop_width <= 0 or length - loop_width <= 0:
        raise ValueError("Loop width bigger than world width. Impossible.")
    
    arr = np.ones((length, width)).astype(int)

    # Top
    arr[:loop_width, :] = 0

    # Bottom
    arr[length-loop_width:, :] = 0

    # Left
    arr[:, :loop_width] = 0

    # Right
    arr[:, width-loop_width:] = 0
    
    return arr_to_worldstr(arr)


def create_two_room_loop_world(room_width, room_length,
                               hallway_width, hallway_length,
                               loop_width):
    if room_width - loop_width <= 0 or room_length - loop_width <= 0:
        raise ValueError("Loop width bigger than world width. Impossible.")
    
    arr = np.ones((room_length,
                   room_width * 2 + hallway_width,)).astype(int)
    # Left Room
    arr[:, :room_width] = 0

    # Right Room
    arr[:, room_width + hallway_width : room_width*2 + hallway_width] = 0
    
    # Hallway
    room_mid = room_length // 2    
    if hallway_length == 1:
        arr[room_mid : room_mid + hallway_length,
            room_width : room_width + hallway_width] = 0
    else:
        hallway_mid = hallway_length // 2
        arr[room_mid - hallway_mid : room_mid + hallway_mid,
            room_width : room_width + hallway_width] = 0

    # Left room loop
    arr[loop_width:room_length-loop_width, loop_width:room_width-loop_width] = 1

    # Right room loop
    arr[loop_width:room_length-loop_width,
        room_width+hallway_width+loop_width:room_width+hallway_width+room_width-loop_width] = 1
    
    return arr_to_worldstr(arr)


def create_free_world(width, length):
    arr = np.zeros((length, width)).astype(int)
    return arr_to_worldstr(arr)


def create_hallway_world(main_width, main_length,
                         branch_width=1, branch_length=3,
                         num_branches=3):
    arr = np.ones((branch_length*2+main_length,
                   main_width)).astype(int)

    # main hallway
    arr[branch_length:branch_length+main_length, :] = 0

    # branches
    gap = main_width // (num_branches*branch_width)
    for i in range(num_branches):
        arr[:, i*(gap + branch_width): i*(gap + branch_width) + branch_width] = 0
        
    return arr_to_worldstr(arr)


def create_connected_hallway_world(main_width, main_length,
                                   branch_width=1, branch_length=3,
                                   num_branches=3):
    arr = np.ones((branch_length*2+main_length,
                   main_width)).astype(int)

    # main hallways
    arr[branch_length:branch_length+main_length, :] = 0
    arr[:main_length, :] = 0
    arr[branch_length*2:branch_length*2+main_length, :] = 0        

    # branches
    gap = main_width // (num_branches*branch_width)
    for i in range(num_branches):
        arr[:, i*(gap + branch_width): i*(gap + branch_width) + branch_width] = 0
        
    return arr_to_worldstr(arr)
