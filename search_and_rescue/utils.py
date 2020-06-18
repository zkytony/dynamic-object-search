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
