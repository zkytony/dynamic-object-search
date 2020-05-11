# Motion policy

import pomdp_py
import random
from ...domain.state import *

class IterativeMotionPolicy(pomdp_py.GenerativeDistribution):
    """A simple Deterministic motion policy where the object
    basically iterates through a given list of
    points. First in forward order, then in
    reversed order."""
    def __init__(self, ordered_points, epsilon_path=1e-9, epsilon=1e-9, loop=True):
        self._ordered_points = ordered_points
        if loop:
            self._ordered_points.extend(reversed(ordered_points[1:]))
        self._points_set = set(ordered_points)
        self._epsilon = epsilon
        self._epsilon_path = epsilon_path  # TODO: THIS IS TEMPORARY

    def random(self, object_state):
        return self.argmax(object_state)

    def argmax(self, object_state):
        next_pose, _ = self._next_pose(object_state)
        return ObjectState(object_state['id'],
                           object_state.objclass,
                           next_pose,
                           time=object_state.time+1)

    def _next_pose(self, object_state):
        if object_state.time < 0:
            raise ValueError("Unexpected. Static object should not have motion policy")
        if object_state.pose in self._points_set:
            index = object_state.time % len(self._ordered_points)
            next_index = self.next_index(index)
            return self._ordered_points[next_index], next_index
        else:
            return (-1,-1), -1

    def next_index(self, index):
        return (index + 1) % len(self._ordered_points)

    def random_index(self):
        return random.randint(0, len(self._ordered_points)-1)

    def probability(self, next_object_state, cur_object_state):
        if next_object_state.pose not in self._points_set\
           or cur_object_state.pose not in self._points_set:
            return self._epsilon
        
        next_i = next_object_state.time % len(self._ordered_points)
        cur_i = cur_object_state.time % len(self._ordered_points)
        # If the indices are correct
        if next_i == self.next_index(cur_i):
            # If the poses match with the pose list in this motion policy
            if next_object_state["pose"] == self._ordered_points[next_i]\
               and cur_object_state["pose"] == self._ordered_points[cur_i]:
                # Then it is a high probability
                return 1.0 - self._epsilon_path
        # Otherwise, it is a low probability.
        return self._epsilon_path


class RandomStayPolicy(pomdp_py.GenerativeDistribution):
    
    def __init__(self, grid_map, pr_stay=0.3):
        self._pr_stay = pr_stay
        self._grid_map = grid_map
        self._actions = {(0,1), (0,-1), (1,0), (-1,0)}

    def probability(self, next_object_state, cur_object_state):
        cur_object_pose = cur_object_state.pose
        next_object_pose = next_object_state.pose        
        diff_x = abs(cur_object_pose[0] - next_object_pose[0])
        diff_y = abs(cur_object_pose[1] - next_object_pose[1])
        if not ((diff_x == 1 and diff_y == 0)
                or (diff_x == 0 and diff_y == 1)
                or (diff_x == 0 and diff_y == 0)):
            return 1e-9
        
        if cur_object_pose == next_object_pose:
            return self._pr_stay
        candidate_actions = self._compute_candidate_actions(cur_object_state)
        if len(candidate_actions) == 0:
            return 1.0 - self._pr_stay
        else:
            return (1.0 - self._pr_stay) / len(candidate_actions)

    def _compute_candidate_actions(self, object_state):
        candidate_actions = []
        for action in self._actions:
            next_object_pose = (object_state.pose[0] + action[0],
                                object_state.pose[1] + action[1])
            if self._grid_map.within_bounds(next_object_pose)\
               and next_object_pose not in self._grid_map.obstacle_poses:
                candidate_actions.append((action, next_object_pose))
        return candidate_actions

    def random(self, object_state, argmax=False):
        if argmax:
            move = self._pr_stay < 0.5
        else:
            move = random.uniform(0,1) > self._pr_stay
        if move:
            # move
            candidate_actions = self._compute_candidate_actions(object_state)
            action, next_object_pose = random.choice(candidate_actions)
            return ObjectState(object_state['id'],
                               object_state.objclass,
                               next_object_pose,
                               time=object_state.time+1)
        else:
            # stay
            return ObjectState(object_state['id'],
                               object_state.objclass,
                               object_state.pose,
                               time=object_state.time+1)

    def argmax(self, object_state):
        return self.random(object_state, argmax=True)
