"""This file has some examples of world string."""
import random
# from .model.components.motion_policy import *

############# Example Worlds ###########
# See env.py:interpret for definition of
# the format

world0 = (
"""
rx...
.x.xT
.....
""", "r")

world1 = (
"""
rx.T...
.x.....
...xx..
.......
.xxx.T.
.xxx...
.......
""", "r")

# Used to test the shape of the sensor
world2 = (
"""
.................
.................
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxTxxxx..
..xxxxxxrxTxxxx..
..xxxxxxxxTxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
.................
.................
""", "r")    

# Used to test sensor occlusion
world3 = (
"""
.................
.................
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxTxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxx...xxxxxx..
..xxxx..xx.xxxx..
..xxxx..r.Txxxx..
..xxxx..xx.xxxx..
..xxxxxx..xxxxx..
..xxxxTx..xxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
.................
.................
""", "r")

world_becky = (
"""
...................x
...................x
...................x
...................x
...........T.......x
...................x
...................x
...................x
...................x
...................x
............r.......
....................
...T................
....................
....................
...T................
....................
....................
....................
....................
""", "r")

###################################################

def random_world(width, length, num_obj, num_obstacles,
                 robot_char="r"):
    worldstr = [[ "." for i in range(width)] for j in range(length)]
    # First place obstacles
    num_obstacles_placed = 0
    while num_obstacles_placed < num_obstacles:
        x = random.randrange(0, width)
        y = random.randrange(0, length)
        if worldstr[y][x] == ".":
            worldstr[y][x] = "x"
            num_obstacles_placed += 1
            
    num_obj_placed = 0
    while num_obj_placed < num_obj:
        x = random.randrange(0, width)
        y = random.randrange(0, length)
        if worldstr[y][x] == ".":
            worldstr[y][x] = "T"
            num_obj_placed += 1

    # Finally place the robot
    while True:
        x = random.randrange(0, width)
        y = random.randrange(0, length)
        if worldstr[y][x] == ".":
            worldstr[y][x] = robot_char
            break

    # Create the string.
    finalstr = []
    for row_chars in worldstr:
        finalstr.append("".join(row_chars))
    finalstr = "\n".join(finalstr)
    return finalstr, robot_char

def place_object(worldstr, x, y, object_char="r"):
    lines = []
    for line in worldstr.split("\n"):
        if len(line) > 0:
            lines.append(line)
    
    width = len(lines[0])
    length = len(lines)
    lines[y] = lines[y][:x] + object_char + lines[y][x+1:]
    return "\n".join(lines)

def place_objects(worldstr, obj_dict):
    for obj_char in obj_dict:
        x, y = obj_dict[obj_char]
        worldstr = place_object(worldstr, x, y, object_char=obj_char)
    return worldstr
