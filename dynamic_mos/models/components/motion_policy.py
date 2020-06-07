# Motion policy

import pomdp_py
import random
import math
from ...domain.state import *
from ...domain.action import *
from ...utils import *

def next_pose(pose, action, motion_scheme="xy"):
    if len(pose) == 2:
        return (pose[0] + action[0],
                pose[1] + action[1])
    elif len(pose) == 3:
        if motion_scheme == "xy":
            return (pose[0] + action[0],
                    pose[1] + action[1],
                    action[2])
        else:
            rx, ry, rth = pose
            forward, angle = action.motion
            rth += angle  # angle (radian)
            rx = int(round(rx + forward*math.cos(rth)))
            ry = int(round(ry + forward*math.sin(rth)))
            rth = rth % (2*math.pi)
            return (rx, ry, rth)
        

class IterativeMotionPolicy(pomdp_py.GenerativeDistribution):
    """A simple Deterministic motion policy where the object
    basically iterates through a given list of
    points. First in forward order, then in
    reversed order."""
    def __init__(self, ordered_points,
                 epsilon_path=1e-9, epsilon=1e-9, loop=True):
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


class StochaisticPolicy(pomdp_py.GenerativeDistribution):
    def __init__(self, grid_map, motion_actions=None):
        self._grid_map = grid_map
        if motion_actions is None:
            self._motion_actions = create_motion_actions(scheme="xy")
        else:
            self._motion_actions = motion_actions
        
        # Compute a map from position to candidate motions
        # for all positions on the map (for efficiency)
        self._legal_actions = {}
        for x in range(grid_map.width):
            for y in range(grid_map.length):
                if (x,y) not in grid_map.obstacle_poses:
                    self._legal_actions[(x,y)] = self._compute_legal_actions((x,y))

    def _compute_legal_actions(self, object_pose):
        legal_actions = []
        for action in self._motion_actions:
            next_object_pose = next_pose(object_pose, action.motion)
            if self._grid_map.within_bounds(next_object_pose)\
               and next_object_pose not in self._grid_map.obstacle_poses:
                legal_actions.append(action)
        return legal_actions

    @property
    def motion_actions(self):
        return self._motion_actions

    @property
    def legal_actions(self):
        return self._legal_actions 

    
class RandomStayPolicy(StochaisticPolicy):
    """With probability `pr_stay` the object stays and with the
    complement of that probability, the object moves randomly"""
    
    def __init__(self, grid_map, pr_stay=0.3):
        self._pr_stay = pr_stay
        super().__init__(grid_map)

    def probability(self, next_object_state, cur_object_state):
        cur_object_pose = cur_object_state.pose
        next_object_pose = next_object_state.pose        
        diff_x = abs(cur_object_pose[0] - next_object_pose[0])
        diff_y = abs(cur_object_pose[1] - next_object_pose[1])
        if not ((diff_x == STEP_SIZE and diff_y == 0)
                or (diff_x == 0 and diff_y == STEP_SIZE)
                or (diff_x == 0 and diff_y == 0)):
            return 1e-9
        
        if cur_object_pose == next_object_pose:
            return self._pr_stay
        legal_actions = self._legal_actions[cur_object_pose]
        if len(legal_actions) == 0:
            return 1.0 - self._pr_stay
        else:
            return (1.0 - self._pr_stay) / len(legal_actions)

    def random(self, object_state, argmax=False):
        if argmax:
            move = self._pr_stay < 0.5
        else:
            move = random.uniform(0,1) > self._pr_stay
        if move:
            # move
            legal_actions = self._legal_actions[object_state.pose]
            action = random.choice(legal_actions)
            next_object_pose = next_pose(object_state.pose, action.motion)          
        else:
            # stay
            next_object_pose = object_state.pose
        return ObjectState(object_state['id'],
                           object_state.objclass,
                           next_object_pose,
                           time=object_state.time+1)

    def argmax(self, object_state):
        return self.random(object_state, argmax=True)


class EpsilonGoalPolicy(StochaisticPolicy):
    def __init__(self, grid_map, goal_pose, epsilon=0.1, stop_at_goal=False):
        super().__init__(grid_map)
        self._epsilon = epsilon
        self._goal_pose = goal_pose
        self._stop_at_goal = stop_at_goal

        # Compute a greedy policy (based on Dijkstra) to
        # arrive at the goal from any location from the map
        self._policy = {}  # maps from position to next position
        for x in range(self._grid_map.width):
            for y in range(self._grid_map.length):
                if (x,y) not in self._policy and (x,y) not in self._grid_map.obstacle_poses:
                    path = self._grid_map.path_between((x,y), self._goal_pose,
                                                       self._motion_actions,
                                                       return_actions=False)
                    assert path[0] == (x,y)
                    self._policy[(x,y)] = path[0]
                    for i in range(1, len(path)):
                        self._policy[path[i-1]] = path[i]

    @property
    def goal_pose(self):
        return self._goal_pose

    def probability(self, next_object_state, cur_object_state):
        cur_object_pose = cur_object_state.pose
        next_object_pose = next_object_state.pose        
        diff_x = abs(cur_object_pose[0] - next_object_pose[0])
        diff_y = abs(cur_object_pose[1] - next_object_pose[1])
        if not ((diff_x == STEP_SIZE and diff_y == 0)
                or (diff_x == 0 and diff_y == STEP_SIZE)
                or (diff_x == 0 and diff_y == 0)):
            return 1e-9
        # Once reached goal, won't move.
        if self._stop_at_goal and cur_object_pose == self._goal_pose:
            if next_object_pose != cur_object_pose:
                return 1e-9
            else:
                return 1.0 - 1e-9
        
        expected_next_pose = self._policy[cur_object_pose]
        if next_object_pose != expected_next_pose:
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def random(self, object_state, argmax=False):
        if self._stop_at_goal and object_state.pose == self._goal_pose:
            # Once reached goal, won't move.
            next_object_pose = object_state.pose
        else:
            # Have not reached goal
            if random.uniform(0,1) > self._epsilon:
                # Moves according to policy
                next_object_pose = self._policy[object_state.pose]
            else:
                # Moves randomly
                legal_actions = self._legal_actions[object_state.pose]
                action = random.choice(legal_actions)            
                next_object_pose = next_pose(object_state.pose, action.motion)
        return ObjectState(object_state["id"],
                           object_state.objclass,
                           next_object_pose,
                           time=object_state.time+1)


class AdversarialPolicy(StochaisticPolicy):
    def __init__(self, grid_map, sensor_range, pr_stay=0.3,
                 rule="avoid", motion_actions=None):
        """With probability `pr_stay`, the object stays in place.
        With the complement probability, the agent moves to maintain
        a distance of more than sensor_range+1 from the robot. The
        object randomly chooses an action that suffices maintaining
        that distance.

        rule can be `avoid` or `keep`. If `avoid`, the object will
        always move away from the robot. If `keep`, the object will
        choose actions that maintains sensor range distance away
        from the robot."""
        super().__init__(grid_map, motion_actions=motion_actions)
        self._sensor_range = sensor_range
        self._pr_stay = pr_stay
        self._rule = rule

    def _adversarial_actions(self, object_pose, robot_pose):
        """Maintain distance if possible. If not at all,
        then just return the legal actions at this object pose"""
        candidate_actions = []        
        if self._rule == "keep":
            for action in self._legal_actions[object_pose]:
                next_dist = euclidean_dist(next_pose(object_pose, action.motion), robot_pose)
                if next_dist > self._sensor_range*math.sqrt(2):
                    candidate_actions.append(action)
        elif self._rule == "avoid":
            cur_dist = euclidean_dist(robot_pose, object_pose)
            for action in self._legal_actions[object_pose]:
                next_dist = euclidean_dist(next_pose(object_pose, action.motion), robot_pose)
                if next_dist > cur_dist:
                    candidate_actions.append(action)
        elif self._rule == "chase":
            # Here the object is actually chasing the robot (useful if
            # you want to use this to model a robot's policy (robot/object swap)
            cur_dist = euclidean_dist(robot_pose, object_pose)
            for action in self._legal_actions[object_pose[:2]]:
                next_dist = euclidean_dist(next_pose(object_pose[:2], action.motion), robot_pose)
                if next_dist < cur_dist:
                    candidate_actions.append(action)
        else:
            raise ValueError("Unknown adversarial rule %s" % self._rule)
        return candidate_actions
        
    def probability(self, next_object_state, cur_object_state, cur_robot_state):
        cur_object_pose = cur_object_state.pose
        next_object_pose = next_object_state.pose        
        diff_x = abs(cur_object_pose[0] - next_object_pose[0])
        diff_y = abs(cur_object_pose[1] - next_object_pose[1])
        if not ((diff_x == STEP_SIZE and diff_y == 0)
                or (diff_x == 0 and diff_y == STEP_SIZE)
                or (diff_x == 0 and diff_y == 0)):
            return 1e-9

        actions = self._adversarial_actions(cur_object_pose, cur_robot_state.pose)
        if len(actions) == 0:
            # No adversarial actions possible.
            if cur_object_pose == next_object_pose:
                # Either the object decides to stay, or it has no clue. Either way,
                # this is basically the only thing that can happen
                return 1.0 - 1e-9
            else:
                return 1e-9
        else:
            # There are candidate actions
            if next_object_pose == cur_object_pose:
                # But the object chose to stay
                return self._pr_stay
            else:
                # The object must have taken an adversarial action
                for action in actions:
                    if next_pose(cur_object_pose, action.motion) == next_object_pose:
                        return (1.0 - self._pr_stay) / len(actions)
                return 1e-9

    def random(self, object_state, robot_state):
        if random.uniform(0,1) > self._pr_stay:
            # move adversarially
            candidate_actions = self._adversarial_actions(object_state.pose, robot_state.pose)
            if len(candidate_actions) == 0:
                # won't move, because no adversarial action
                next_object_pose = object_state.pose
            else:
                # randomly choose an action from candidates
                action = random.choice(candidate_actions)
                next_object_pose = next_pose(object_state.pose, action.motion)
        else:
            # stay
            next_object_pose = object_state.pose

        if isinstance(object_state, RobotState):
            # here is the adversarial agent
            assert isinstance(robot_state, ObjectState)
            time = robot_state.time + 1
        else:
            time = object_state.time + 1
            
        return ObjectState(object_state["id"],
                           object_state.objclass,
                           next_object_pose,
                           time=time)


class AdverserialGoalPolicy(AdversarialPolicy, EpsilonGoalPolicy):
    def __init__(self, grid_map, goal_pose, sensor_range, pr_stay=0.1, stop_at_goal=False):
        AdversarialPolicy.__init__(self, grid_map, sensor_range, pr_stay=pr_stay)
        EpsilonGoalPolicy.__init__(self, grid_map, goal_pose, epsilon=1e-9, stop_at_goal=stop_at_goal)

    def probability(self, next_object_state, cur_object_state, cur_robot_state):
        cur_object_pose = cur_object_state.pose
        next_object_pose = next_object_state.pose        
        diff_x = abs(cur_object_pose[0] - next_object_pose[0])
        diff_y = abs(cur_object_pose[1] - next_object_pose[1])
        if not ((diff_x == STEP_SIZE and diff_y == 0)
                or (diff_x == 0 and diff_y == STEP_SIZE)
                or (diff_x == 0 and diff_y == 0)):
            return 1e-9
        # TODO: This needs more work.
