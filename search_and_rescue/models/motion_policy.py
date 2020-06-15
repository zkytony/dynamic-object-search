import math
import pomdp_py
import random
import copy
from search_and_rescue.env.action import *
from search_and_rescue.env.state import *

def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

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

    def valid_motions(self, robot_id, robot_pose, all_motion_actions):
        """
        Returns a set of MotionAction(s) that are valid to
        be executed from robot pose (i.e. they will not bump
        into obstacles). The validity is determined under
        the assumption that the robot dynamics is deterministic.
        """
        valid = set({})
        for motion_action in all_motion_actions:
            if not isinstance(motion_action, MotionAction):
                raise ValueError("This (%s) is not a motion action" % str(motion_action))

            next_robot_pose = next_pose(robot_pose, motion_action)
            if (next_robot_pose[:2] in self._grid_map.obstacle_poses)\
               or (not self._grid_map.within_bounds(next_robot_pose)):
                next_robot_pose = robot_pose
                
            if next_robot_pose != robot_pose:
                # robot moved --> valid motion
                valid.add(motion_action)

        if Stay in all_motion_actions:
            valid.add(Stay)
                
        return valid

    def get_neighbors(self, pose, all_motion_actions):
        neighbors = {}
        if len(pose) == 2:
            pose = (pose[0], pose[1], 0)
        for motion_action in all_motion_actions:
            if not isinstance(motion_action, MotionAction):
                raise ValueError("This (%s) is not a motion action" % str(motion_action))

            next_robot_pose = next_pose(pose, motion_action)
            if (next_robot_pose[:2] not in self._grid_map.obstacle_poses)\
               and (self._grid_map.within_bounds(next_robot_pose)):
                neighbors[next_robot_pose[:2]] = motion_action
        return neighbors
    
    def path_between(self, position1, position2, all_motion_actions, return_actions=True):
        """Note that for the return_actions=True case to return reasonable
        actions, the motion actions scheme needs to be `xy`, i.e. along the axes"""
        # Finds a path between position1 and position2.
        # Uses the Dijkstra's algorithm.
        V = set({(x,y)    # all valid positions
                 for x in range(self._grid_map.width) 
                 for y in range(self._grid_map.length)
                 if self._grid_mapwithin_bounds((x,y)) and (x,y) not in self._grid_map.obstacle_poses})
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
        

class BasicMotionPolicy(StochaisticPolicy):
    """Adversarial Target motion policy."""
    def __init__(self, object_id, grid_map, motion_actions):
        self._object_id = object_id
        self._motion_actions = motion_actions
        super().__init__(grid_map, motion_actions)

    def probability(self, next_object_state, state, action):
        if next_object_state["id"] != self._object_id:
            return 1e-9
        
        legal_actions = self._legal_actions[state.pose(self._object_id)[:2]]
        if action not in legal_actions:
            return 1e-9
        else:
            if next_pose(state.pose(next_object_state["id"]), action.motion)\
               == next_object_state.pose:
                return 1.0 - 1e-9
            else:
                return 1e-9

    def random(self, state, action):
        """Given MosOOState, return ObjectState"""
        next_object_state = copy.deepcopy(state.object_states[self._object_id])        
        if action not in self._legal_actions[state.pose(self._object_id)]:
            # raise ValueError("Action %s cannot be taken in state %s" % (str(action), str(state)))
            # Action is not lega. Does not move.
            return next_object_state

        cur_object_pose = state.pose(self._object_id)
        next_object_pose = next_pose(cur_object_pose, action.motion)
        next_object_state["pose"] = next_object_pose
        return next_object_state


class AdversarialPolicy(StochaisticPolicy):
    def __init__(self, object_id, robot_id,
                 grid_map, sensor_range, pr_stay=0.3,
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
        self._object_id = object_id  # the adversary
        self._robot_id = robot_id  # the chaser
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
        
    def probability(self, next_object_state, state):
        cur_object_pose = state.pose(self._object_id)
        next_object_pose = next_object_state.pose
        diff_x = abs(cur_object_pose[0] - next_object_pose[0])
        diff_y = abs(cur_object_pose[1] - next_object_pose[1])
        if not ((diff_x == STEP_SIZE and diff_y == 0)
                or (diff_x == 0 and diff_y == STEP_SIZE)
                or (diff_x == 0 and diff_y == 0)):
            return 1e-9

        actions = self._adversarial_actions(cur_object_pose, state.pose(self._robot_id))
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

    def random(self, state):
        object_state = state.object_states[self._object_id]
        robot_state = state.object_states[self._robot_id]        
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

        next_object_state = copy.deepcopy(object_state)
        next_object_state["pose"] = next_object_pose
        next_object_state["time"] = object_state.time + 1
        return next_object_state


# UNIT TESTS
def unittest():
    from search_and_rescue.models.grid_map import GridMap
    grid_map = GridMap(10, 10,
                       {0: (2,3), 5:(4,9)})
    motion_actions = create_motion_actions(can_stay=True)
    
    bmp = BasicMotionPolicy(4, grid_map, motion_actions)
    next_object_state = SearcherState(4, (4,5), (), True)
    state = JointState({3: VictimState(3, (1,0), True),
                        4: SearcherState(4, (4,4), (), True)})
    action = MoveSouth
    print(bmp.probability(next_object_state, state, action))
    print(bmp.random(state, action))    

    adv = AdversarialPolicy(3, 4, grid_map, 3, motion_actions=motion_actions)
    print(adv.probability(next_object_state, state))
    print(adv.random(state))    

if __name__ == '__main__':
    unittest()
