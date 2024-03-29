import math
import pomdp_py
import random
import copy
from search_and_rescue.env.action import *
from search_and_rescue.env.state import *
from search_and_rescue.utils import *

def next_pose(pose, action, motion_scheme="xy"):
    if len(pose) == 2:
        return (pose[0] + action[0],
                pose[1] + action[1])
    elif len(pose) == 3:
        if motion_scheme == "xy":
            if action[2] == -1:
                th = pose[2]
            else:
                th = action[2]
            return (pose[0] + action[0],
                    pose[1] + action[1],
                    th)
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
    def grid_map(self):
        return self._grid_map

    @property
    def legal_actions(self):
        return self._legal_actions

    def valid_motions(self, robot_pose, all_motion_actions):
        """
        Returns a set of MotionAction(s) that are valid to
        be executed from robot pose (i.e. they will not bump
        into obstacles). The validity is determined under
        the assumption that the robot dynamics is deterministic.
        """
        valid = set(self._legal_actions[robot_pose[:2]])
        if Stay in all_motion_actions:
            valid.add(Stay)
        # valid = set({})
        # for motion_action in all_motion_actions:
        #     if not isinstance(motion_action, MotionAction):
        #         raise ValueError("This (%s) is not a motion action" % str(motion_action))

        #     next_robot_pose = next_pose(robot_pose, motion_action.motion)
        #     if (next_robot_pose[:2] in self._grid_map.obstacle_poses)\
        #        or (not self._grid_map.within_bounds(next_robot_pose)):
        #         next_robot_pose = robot_pose
                
        #     if next_robot_pose != robot_pose:
        #         # robot moved --> valid motion
        #         valid.add(motion_action)

        # if Stay in all_motion_actions:
        #     valid.add(Stay)
        return valid

    def get_neighbors(self, pose, all_motion_actions):
        neighbors = {}
        if len(pose) == 2:
            pose = (pose[0], pose[1], 0)
        for motion_action in all_motion_actions:
            if not isinstance(motion_action, MotionAction):
                raise ValueError("This (%s) is not a motion action" % str(motion_action))

            next_robot_pose = next_pose(pose, motion_action.motion)
            if (next_robot_pose[:2] not in self._grid_map.obstacle_poses)\
               and (self._grid_map.within_bounds(next_robot_pose)):
                
                neighbors[next_robot_pose] = motion_action
        return neighbors
    
    def path_between(self, position1, position2, all_motion_actions, return_actions=True):
        """Note that for the return_actions=True case to return reasonable
        actions, the motion actions scheme needs to be `xy`, i.e. along the axes"""
        # Finds a path between position1 and position2.
        # Uses the Dijkstra's algorithm.
        V = set({(x,y)    # all valid positions
                 for x in range(self._grid_map.width) 
                 for y in range(self._grid_map.length)
                 if self._grid_map.within_bounds((x,y)) and (x,y) not in self._grid_map.obstacle_poses})
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
                if d[v] + cost_vw < d[w[:2]]:
                    d[w[:2]] = d[v] + cost_vw
                    prev[w[:2]] = (v, motion_action)

        # Return a path
        path = []
        pair = prev[position2[:2]]
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
    """Basic motion policy; No restriction on what motion action can be taken
    other than being valid in the map."""
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
        assert isinstance(action, MotionAction),\
            "Motion policy only handles motion action"
        next_object_state = copy.deepcopy(state.object_states[self._object_id])
        if action not in self._legal_actions[state.pose(self._object_id)[:2]]:
            # raise ValueError("Action %s cannot be taken in state %s" % (str(action), str(state)))
            # Action is not lega. Does not move.
            return next_object_state

        cur_object_pose = state.pose(self._object_id)
        next_object_pose = next_pose(cur_object_pose, action.motion)
        next_object_state["pose"] = next_object_pose
        return next_object_state


class AdversarialPolicy(StochaisticPolicy):
    def __init__(self, adv_id, agent_id,
                 grid_map, sensor_range, pr_stay=1e-9,
                 rule="avoid", motion_actions=None):
        """With probability `pr_stay`, the object stays in place.
        With the complement probability, the agent moves to maintain
        a distance of more than sensor_range+1 from the agent. The
        object randomly chooses an action that suffices maintaining
        that distance.

        rule can be `avoid` or `keep`. If `avoid`, the object will
        always move away from the agent. If `keep`, the object will
        choose actions that maintains sensor range distance away
        from the agent."""
        super().__init__(grid_map, motion_actions=motion_actions)
        self._adv_id = adv_id  # the adversary
        self._agent_id = agent_id  # the chaser
        self._sensor_range = sensor_range
        self._pr_stay = pr_stay
        self._rule = rule

    def _adversarial_actions(self, adv_pose, agent_pose):
        """Maintain distance if possible. If not at all,
        then just return the legal actions at this adv pose"""
        candidate_actions = []
        assert len(adv_pose) == len(agent_pose)
        assert len(adv_pose) == 2
        if self._rule == "keep":
            for action in self._legal_actions[adv_pose]:
                next_dist = euclidean_dist(next_pose(adv_pose, action.motion)[:2], agent_pose)
                if next_dist > self._sensor_range*math.sqrt(2):
                    candidate_actions.append(action)
        elif self._rule == "avoid":
            cur_dist = euclidean_dist(agent_pose, adv_pose)
            for action in self._legal_actions[adv_pose]:
                next_dist = euclidean_dist(next_pose(adv_pose, action.motion)[:2], agent_pose)
                if next_dist > cur_dist:
                    candidate_actions.append(action)
        elif self._rule == "chase":
            # Here the adv is actually chasing the agent (useful if
            # you want to use this to model a agent's policy (agent/adv swap)
            cur_dist = euclidean_dist(agent_pose, adv_pose)
            for action in self._legal_actions[adv_pose]:
                next_dist = euclidean_dist(next_pose(adv_pose, action.motion)[:2], agent_pose)
                if next_dist < cur_dist:
                    candidate_actions.append(action)
        else:
            raise ValueError("Unknown adversarial rule %s" % self._rule)
        return candidate_actions
        
    def probability(self, next_adv_state, state, agent_state=None):
        # The reason we have to do this check is because of
        # histogram belief update; You cannot pass in full JointState
        # and can only pass in adv state.
        if not isinstance(state, JointState):
            cur_adv_pose = state.pose
            agent_pose = agent_state.pose
        else:
            cur_adv_pose = state.pose(self._adv_id)
            agent_pose = state.pose(self._agent_id)
        next_adv_pose = next_adv_state.pose
        diff_x = abs(cur_adv_pose[0] - next_adv_pose[0])
        diff_y = abs(cur_adv_pose[1] - next_adv_pose[1])
        if not ((diff_x == STEP_SIZE and diff_y == 0)
                or (diff_x == 0 and diff_y == STEP_SIZE)
                or (diff_x == 0 and diff_y == 0)):
            return 1e-9

        actions = self._adversarial_actions(cur_adv_pose[:2], agent_pose[:2])
        if len(actions) == 0:
            # No adversarial actions possible.
            if cur_adv_pose == next_adv_pose:
                # Either the adv decides to stay, or it has no clue. Either way,
                # this is basically the only thing that can happen
                return 1.0 - 1e-9
            else:
                return 1e-9
        else:
            # There are candidate actions
            if next_adv_pose == cur_adv_pose:
                # But the adv chose to stay
                return self._pr_stay
            else:
                # The adv must have taken an adversarial action
                for action in actions:
                    if next_pose(cur_adv_pose, action.motion) == next_adv_pose:
                        return (1.0 - self._pr_stay) / len(actions)
                return 1e-9

    def random(self, state, get_action=False):
        adv_state = state.object_states[self._adv_id]
        agent_state = state.object_states[self._agent_id]
        action = None
        if random.uniform(0,1) > self._pr_stay:
            # move adversarially
            candidate_actions = self._adversarial_actions(adv_state.pose[:2],
                                                          agent_state.pose[:2])
            if len(candidate_actions) == 0:
                # won't move, because no adversarial action
                next_adv_pose = adv_state.pose
                action = Stay
            else:
                # randomly choose an action from candidates
                action = random.choice(candidate_actions)
                next_adv_pose = next_pose(adv_state.pose, action.motion)
        else:
            # stay
            next_adv_pose = adv_state.pose
            action = Stay

        next_adv_state = copy.deepcopy(adv_state)
        next_adv_state["pose"] = next_adv_pose
        next_adv_state["time"] = adv_state.time + 1
        if get_action:
            return action
        else:
            return next_adv_state


class MixedPolicy(pomdp_py.GenerativeDistribution):
    def __init__(self, object_id, motion_policies):
        self.object_id = object_id
        self._motion_policies = list(motion_policies)
        self._mpoli_weight = 1.0 / len(self._motion_policies)   # Pr(mpoli | state)

    def probability(self, next_object_state, state, **kwargs):
        prob = 0
        for mpoli in self._motion_policies:
            prob += mpoli.probability(next_object_state, state, **kwargs) * self._mpoli_weight
        return prob

    def random(self, state, **kwargs):
        # First, pick a motion policy according to weight
        ## Note even though below is just a uniform sampling, we might (in the future)
        ## have non-uniform weights for the policies.
        mpoli_chosen = random.choices(
            self._motion_policies,
            weights=[self._mpoli_weight]*len(self._motion_policies), k=1)[0]
        return mpoli_chosen.random(state, **kwargs)



# UNIT TESTS
def unittest():
    from search_and_rescue.models.grid_map import GridMap
    grid_map = GridMap(10, 10,
                       {0: (9,9), 5:(4,9)})
    motion_actions = create_motion_actions(can_stay=True)
    
    bmp = BasicMotionPolicy(4, grid_map, motion_actions)
    next_object_state = SearcherState(4, (4,5), (), True)
    state = JointState({3: VictimState(3, (1,0), (), True),
                        4: SearcherState(4, (4,4), (), True)})
    action = MoveSouth
    print(bmp.probability(next_object_state, state, action))
    print(bmp.random(state, action))    

    adv = AdversarialPolicy(3, 4, grid_map, 3, motion_actions=motion_actions)
    print(adv.probability(next_object_state, state))
    print(adv.random(state))

    start_position = (0,0,0)
    end_position = (5,5,math.pi*2/3)
    path = bmp.path_between(start_position, end_position, motion_actions)
    print(path)

    print("Create a situation to test out \"probability\" function of AdversarialPolicy")
    searcher_id = 3
    suspect_id = 4
    
    searcher_pose = (3,3, math.pi*2/3)
    suspect_pose = (2,5, math.pi)    
    state = JointState({searcher_id: SearcherState(searcher_id, (*searcher_pose,), (), True),
                        suspect_id: SuspectState(suspect_id, (*suspect_pose,), (), True)})

    searcher_pose = next_pose(searcher_pose, MoveWest.motion)
    next_state = JointState({searcher_id: SearcherState(searcher_id, (*searcher_pose,), (), True),
                             suspect_id: SuspectState(suspect_id, (*suspect_pose,), (), True)})
    
    searcher_adv_policy = AdversarialPolicy(searcher_id, suspect_id, grid_map, None,
                                            rule="chase", motion_actions=motion_actions)
    print(searcher_adv_policy.probability(next_state.object_states[searcher_id],
                                          state.object_states[searcher_id],
                                          agent_state=state.object_states[suspect_id]))

if __name__ == '__main__':
    unittest()
