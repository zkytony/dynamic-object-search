# Motion policy

import pomdp_py
import random
from ...domain.state import *

class IterativeMotionPolicy(pomdp_py.GenerativeDistribution):
    """A simple Deterministic motion policy where the object
    basically iterates through a given list of
    points. First in forward order, then in
    reversed order."""
    def __init__(self, ordered_points, epsilon=1e-9):
        """
        Requires the path contains no loop.
        """
        if len(set(ordered_points)) != len(ordered_points):
            raise ValueError("The path must not contain loop or intersection with itself.")
        self._ordered_points = ordered_points
        self._indices = {
            p:i for i,p in enumerate(self._ordered_points)
        }
        self._epsilon = epsilon

    def random(self, object_state):
        return self.argmax(object_state)

    def argmax(self, object_state):
        next_pose, _ = self._next_pose(object_state)
        return ObjectState(object_state['id'],
                           object_state.objclass,
                           next_pose)

    def _next_pose(self, object_state):
        if object_state.pose in self._indices:
            index = self._indices[object_state.pose]
            next_index = self.next_index(index)
            return self._ordered_points[next_index], next_index
        else:
            return (-1,-1), -1

    def next_index(self, index):
        return (index + 1) % len(self._ordered_points)

    def random_index(self):
        return random.randint(0, len(self._ordered_points)-1)

    def probability(self, next_object_state, cur_object_state):
        if next_object_state.pose not in self._indices\
           or cur_object_state.pose not in self._indices:
            return self._epsilon
        
        next_i = self._indices[next_object_state.pose]
        cur_i = self._indices[cur_object_state.pose]
        # If the indices are correct
        if next_i == self.next_index(cur_i):
            # If the poses match with the pose list in this motion policy
            if next_object_state["pose"] == self._ordered_points[next_i]\
               and cur_object_state["pose"] == self._ordered_points[cur_i]:
                # Then it is a high probability
                return 1.0 - self._epsilon
        # Otherwise, it is a low probability.
        return self._epsilon
