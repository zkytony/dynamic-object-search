"""Optional grid map to assist collision avoidance during planning."""

from search_and_rescue.env.state import *
from search_and_rescue.env.action import *

class GridMap:
    """This map assists the agent to avoid planning invalid
    actions that will run into obstacles. Used if we assume
    the agent has a map. This map does not contain information
    about the object locations."""
    def __init__(self, width, length, obstacles):
        """
        Args:
            obstacle (dict): Map from objid to (x,y); The object is
                                   supposed to be an obstacle.
            width (int): width of the grid map
            length (int): length of the grid map 
        """
        self.width = width
        self.length = length
        self._obstacles = obstacles
        # An MosOOState that only contains poses for obstacles;
        # This is to allow calling RobotTransitionModel.if_move_by
        # function.
        self._obstacle_states = {
            objid: ObstacleState(objid, self._obstacles[objid])
            for objid in self._obstacles
        }
        self._obstacle_poses = set({
            self._obstacles[objid] for objid in self._obstacles
        })
        # Free poses
        self._free_poses = set({
            (x,y)
            for x in range(self.width)
            for y in range(self.length)
            if (x,y) not in self._obstacle_poses
        })

    @property
    def obstacle_poses(self):
        return self._obstacle_poses

    @property
    def free_poses(self):
        return self._free_poses
    
    @property
    def obstacles(self):
        return self._obstacles 

    def within_bounds(self, position):
        if not (position[0] >= 0 and position[0] < self.width\
                and position[1] >= 0 and position[1] < self.length):
            return False
        if position in self.obstacle_poses:
            return False
        return True
