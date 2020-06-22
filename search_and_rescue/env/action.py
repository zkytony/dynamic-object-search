import pomdp_py
import math
import copy

###### Actions ######
class Action(pomdp_py.Action):
    """Mos action; Simple named action."""
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Action(%s)" % self.name

STEP_SIZE=1
class MotionAction(Action):
    ORIENTATIONS = [0, math.pi, 3*math.pi/2, math.pi/2]
    EAST = (STEP_SIZE, 0, 0)  # x is horizontal; x+ is right. y is vertical; y+ is down.
    WEST = (-STEP_SIZE, 0, math.pi)
    NORTH = (0, -STEP_SIZE, 3*math.pi/2)
    SOUTH = (0, STEP_SIZE, math.pi/2)
    STAY = (0,0,-1)

    def __init__(self, motion, distance_cost=STEP_SIZE, motion_name=None, observing=False):
        """
        motion (tuple): a tuple of floats that describes the motion;
        observing (bool): True if after moving, the agent also observes the environment.
        """
        self.motion = motion
        self.distance_cost = distance_cost
        self.observing = False
        if motion_name is None:
            motion_name = str(motion)
        super().__init__("move-%s" % motion_name)

class LookAction(Action):
    # For simplicity, this LookAction is not parameterized by direction
    def __init__(self, cost=0):
        self.cost = cost
        super().__init__("look")
        
class FindAction(Action):
    def __init__(self):
        super().__init__("find")

Look = LookAction()
Find = FindAction()

MoveEast = MotionAction(MotionAction.EAST, motion_name="East", distance_cost=STEP_SIZE)
MoveWest = MotionAction(MotionAction.WEST, motion_name="West", distance_cost=STEP_SIZE)
MoveNorth = MotionAction(MotionAction.NORTH, motion_name="North", distance_cost=STEP_SIZE)
MoveSouth = MotionAction(MotionAction.SOUTH, motion_name="South", distance_cost=STEP_SIZE)
Stay = MotionAction(MotionAction.STAY, motion_name="Stay", distance_cost=0)

def create_motion_actions(distance_cost=STEP_SIZE, can_stay=False):
    actions = copy.deepcopy({MoveEast, MoveWest, MoveNorth, MoveSouth})
    for a in actions:
        a.distance_cost = distance_cost
    if can_stay:
        return actions | {Stay}
    else:
        return actions    


class ActionCollection(pomdp_py.Action):
    def __init__(self, actions):
        self.actions = actions

    def __getitem__(self, agent_id):
        if agent_id in self.actions:
            return self.actions[agent_id]
        else:
            return None

    def __setitem__(self, agent_id, action):
        self.actions[agent_id] = action

    def __str__(self):
        res = "ActionCollection\n"
        for agent_id in self.actions:
            res += "  %d: %s\n" % (agent_id, str(self.actions[agent_id]))
        return res

    def __repr__(self):
        return "ActionCollection(%s)" % str(self.actions)
    
