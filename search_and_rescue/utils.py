import math

def euclidean_dist(p1, p2):
    if len(p1) != len(p2):
        print("warning: computing distance between two points of different dimensions")
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def to_rad(deg):
    return deg * math.pi / 180.0

def in_range(val, rang):
    # Returns True if val is in range (a,b); Inclusive.
    return val >= rang[0] and val <= rang[1]


#### World building ###
def place_object(worldstr, x, y, object_char="r"):
    lines = []
    for line in worldstr.split("\n"):
        if len(line) > 0:
            lines.append(line)
    
    width = len(lines[0])
    length = len(lines)
    lines[y] = lines[y][:x] + object_char + lines[y][x+1:]
    return "\n".join(lines)

def place_objects(worldstr, obj_poses):
    if type(obj_poses) == dict:
        for obj_char in obj_dict:
            x, y = obj_dict[obj_char]
            worldstr = place_object(worldstr, x, y, object_char=obj_char)
    else:
        for obj_char, obj_pose in obj_poses:
            worldstr = place_object(worldstr, obj_pose[0], obj_pose[1], object_char=obj_char)
    return worldstr
