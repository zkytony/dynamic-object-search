# Motion policy

import pomdp_py
from ...domain.state import *

class IterativeMotionPolicy(pomdp_py.GenerativeDistribution):
    """A simple Deterministic motion policy where the object
    basically iterates through a given list of
    points. First in forward order, then in
    reversed order."""
    def __init__(self, ordered_points, epsilon=1e-9):
        # Create a loop of the ordered points
        self._ordered_points = list(ordered_points) + list(reversed(ordered_points[1:-1]))
        self._epsilon = epsilon

    def random(self, object_state):
        return self.argmax(object_state)

    def argmax(self, object_state):
        next_pose, next_pose_index = self._next_pose(object_state)
        return ObjectState(object_state['id'],
                           object_state.objclass,
                           next_pose,
                           next_pose_index)

    def _next_pose(self, object_state):
        index = object_state['pose_index']
        next_index = (index + 1) % len(self._ordered_points)
        return self._ordered_points[next_index], next_index

    def probability(self, next_object_state, cur_object_state):
        next_i = next_object_state["pose_index"]
        cur_i = cur_object_state["pose_index"]
        # If the indices are correct
        if next_i == (cur_i + 1) % len(self._ordered_points):
            # If the poses match with the pose list in this motion policy
            if next_object_state["pose"] == self._ordered_points[next_i]\
               and cur_object_state["pose"] == self._ordered_points[cur_i]:
                # Then it is a high probability
                return 1.0 - self._epsilon
        # Otherwise, it is a low probability.
        return self._epsilon
