from .example_worlds import place_objects

map1 =\
"""
....
....
xx..
xx..
"""

map2 =\
"""
xxxxxxxxxxxxxxx
xx....xxx....xx
xx...........xx
xx....xxx....xx
xxxxxxxxxxxxxxx
"""

map3 =\
"""
xxxxxxxxxxx
xxr.xxx..xx
xx.......xx
xx..xxx.Dxx
xxxxxxxxxxx
"""

dynamic_world_1 = (place_objects(map1,
                                 {"r": (3,3),
                                  "D": (0,0)}),
                   "r",
                   {"D": [(0,0), (1,0), (2,0)]})

dynamic_world_2 = (place_objects(map2,
                                 {"r": (2,1),
                                  "D": (12,3)}),
                   "r",
                   {"D": [(12,3),(11,3),(11,2),(10,2),(9,2),(8,2),(7,2),(6,2),(5,2),(4,2),(4,1),(3,1)]})


dynamic_world_3 = (place_objects(map3,
                                 {"r": (2,1),
                                  "D": (8,3)}),
                   "r",
                   {"D": [(8,3),(8,2),(7,2),(6,2),(5,2),(4,2),(3,2),(3,1)]})

dynamic_world_4 = (place_objects(map3,
                                 {"r": (2,1),
                                  "D": (8,3),
                                  "E": (3,1)}),
                   "r", {"D": [(8,3),(8,2),(7,2),(6,2),(5,2),(4,2),(3,2),(3,1)],
                         "E": [(3,1),(3,2),(3,3),(2,3),(2,2),(2,1)]}
)

dynamic_world_5 = (place_objects(map2,
                                 {"r": (2,1),
                                  "D": (12,3),
                                  "E": (9,1)}),
                   "r",
                   {"D": [(12,3),(11,3),(11,2),(10,2),(9,2),(8,2),(7,2),(6,2),(5,2),(4,2),(4,1),(3,1)],
                    "E": [(9,1),(9,2),(9,3),(10,3),(11,3),(11,2),(11,1),(10,1)]})

static_world_5 = ((place_objects(map2,
                                 {"r": (2,1),
                                  "D": (12,3),
                                  "E": (9,1)}),
                   "r",
                   {}))

dynamic_world_6 = (place_objects(map2,
                                 {"r": (2,1),
                                  "D": (12,3),
                                  "E": (9,1)}),
                   "r",
                   {"D": ("random", 0.2),
                    "E": ("iterative", [(9,1),(9,2),(9,3),(10,3),(11,3),(11,2),(11,1),(10,1)])})
