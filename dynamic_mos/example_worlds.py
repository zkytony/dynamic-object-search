"""This file has some examples of world string."""
import random

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

dynamic_world_1 = (
"""
D...
....
xx..
xx.r
""", "r", {"D": [(0,0), (1,0), (2,0)]})

dynamic_world_2 = (
"""
xxxxxxxxxxxxxxx
xxr...xxx....xx
xx...........xx
xx....xxx...Dxx
xxxxxxxxxxxxxxx
""", "r", {"D": [(12,3),(11,3),(11,2),(10,2),(9,2),(8,2),(7,2),(6,2),(5,2),(4,2),(4,1),(3,1)]}
)

dynamic_world_3 = (
"""
xxxxxxxxxxx
xxr.xxx..xx
xx.......xx
xx..xxx.Dxx
xxxxxxxxxxx
""", "r", {"D": [(8,3),(8,2),(7,2),(6,2),(5,2),(4,2),(3,2),(3,1)]}
)

dynamic_world_4 = (
"""
xxxxxxxxxxx
xxrExxx..xx
xx.......xx
xx..xxx.Dxx
xxxxxxxxxxx
""", "r", {"D": [(8,3),(8,2),(7,2),(6,2),(5,2),(4,2),(3,2),(3,1)],
           "E": [(3,1),(3,2),(3,3),(2,3),(2,2),(2,1)]}
)

dynamic_world_5 = (
"""
xxxxxxxxxxxxxxx
xxr...xxxE...xx
xx...........xx
xx....xxx...Dxx
xxxxxxxxxxxxxxx
""", "r", {"D": [(12,3),(11,3),(11,2),(10,2),(9,2),(8,2),(7,2),(6,2),(5,2),(4,2),(4,1),(3,1)],
           "E": [(9,1),(9,2),(9,3),(10,3),(11,3),(11,2),(11,1),(10,1)]}
)


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
