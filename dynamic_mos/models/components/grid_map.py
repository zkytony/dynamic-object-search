"""Optional grid map to assist collision avoidance during planning."""

from ..transition_model import RobotTransitionModel
from ...domain.action import *
from ...domain.state import *

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
            objid: ObjectState(objid, "obstacle", self._obstacles[objid])
            for objid in self._obstacles
        }
        self._obstacle_poses = set({
            self._obstacles[objid] for objid in self._obstacles
        })

    @property
    def obstacle_poses(self):
        return self._obstacle_poses
    
    @property
    def obstacles(self):
        return self._obstacles 

    def valid_motions(self, robot_id, robot_pose, all_motion_actions):
        """
        Returns a set of MotionAction(s) that are valid to
        be executed from robot pose (i.e. they will not bump
        into obstacles). The validity is determined under
        the assumption that the robot dynamics is deterministic.
        """
        state = MosOOState(self._obstacle_states)
        state.set_object_state(robot_id,
                               RobotState(robot_id, robot_pose, None, None))

        valid = set({})
        for motion_action in all_motion_actions:
            if not isinstance(motion_action, MotionAction):
                raise ValueError("This (%s) is not a motion action" % str(motion_action))

            next_pose = RobotTransitionModel.if_move_by(robot_id, state,
                                                        motion_action, (self.width, self.length))
            if next_pose != robot_pose:
                # robot moved --> valid motion
                valid.add(motion_action)
        return valid

    def within_bounds(self, position):
        if not (position[0] >= 0 and position[0] < self.width\
                and position[1] >= 0 and position[1] < self.length):
            return False
        if position in self.obstacle_poses:
            return False
        return True

    def get_neighbors(self, pose, all_motion_actions):
        neighbors = {}
        if len(pose) == 2:
            pose = (pose[0], pose[1], 0)
        for motion_action in all_motion_actions:
            if not isinstance(motion_action, MotionAction):
                raise ValueError("This (%s) is not a motion action" % str(motion_action))

            next_pose = RobotTransitionModel.if_move_by(None, None,
                                                        motion_action,
                                                        (self.width, self.length),
                                                        check_collision=False,
                                                        robot_pose=pose)
            if next_pose[:2] not in self._obstacle_poses:
                neighbors[next_pose[:2]] = motion_action
        return neighbors
        
    
    def path_between(self, position1, position2, all_motion_actions, return_actions=True):
        """Note that for the return_actions=True case to return reasonable
        actions, the motion actions scheme needs to be `xy`, i.e. along the axes"""
        # Finds a path between position1 and position2.
        # Uses the Dijkstra's algorithm.
        V = set({(x,y)    # all valid positions
                 for x in range(self.width) 
                 for y in range(self.length)
                 if self.within_bounds((x,y)) and (x,y) not in self._obstacle_poses})
        position1 = position1[:2]  # If it is robot pose then it has length 3.
        S = set({})
        d = {v:float("inf")
             for v in V
             if v != position1}
        d[position1] = 0
        prev = {position1: None}
        while len(S) < len(V):
            diff_set = V - S
            v = min(diff_set, key=lambda v: d[v])
            S.add(v)
            neighbors = self.get_neighbors(v, all_motion_actions)
            for w in neighbors:
                motion_action = neighbors[w]
                cost_vw = motion_action.distance_cost
                if d[v] + cost_vw < d[w]:
                    d[w] = d[v] + cost_vw
                    prev[w] = (v, motion_action)

        # Return a path
        path = []
        pair = prev[position2]
        if pair is None:
            if not return_actions:
                path.append(position2)
        else:
            while pair is not None:
                position, action = pair
                if return_actions:
                    # Return a sequence of actions that moves the robot from position1 to position2.
                    path.append(action)
                else:
                    # Returns the grid cells along the path
                    path.append(position)
                pair = prev[position]
        return list(reversed(path))
        
        
        
