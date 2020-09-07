# This is a simple two-agent domain similar to the tiger domain.
#
# There are two doors, and one person is on one side, and the other
# on the other side.
#
# Each person is (states) either behind the left door, or the right door.
#
# They can both (actions):
#
# - move: move to the other door
# - stay (listen): don't move
# - open-door: open the door in front
#
# (observations) If they stay, they can hear noise, when the other person is at
# the same door as them, or silence if not.
#
# (reward) There is a pursuer and an evader. If when the door is open, and
# both agents are met face to face, then the pursuer wins. Otherwise, the
# evader wins. Whenever a door is open, the game ends.

import pomdp_py

#------State, Action, Observation------#
class GameState(pomdp_py.State):
    def __init__(self, role, side):
        self.role = role
        self.side = side
    def __hash__(self):
        return hash((self.side, self.role))
    def __eq__(self, other):
        if isinstance(other, GameState):        
            return self.role == other.role and self.side == other.side
        return False
    def __str__(self):
        return self.side
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.side)

class PState(GameState):
    def __init__(self, side):
        super().__init__("pursuer", side)
    def copy(self):
        return PState(self.side)        

class EState(GameState):
    def __init__(self, side):
        super().__init__("evader", side)
    def copy(self):
        return EState(self.side)                

class JointState(pomdp_py.State):
    def __init__(self, pstate, estate):
        self.pstate = pstate
        self.estate = estate
    def __hash__(self):
        return hash((self.pstate, self.estate))
    def __eq__(self, other):
        if isinstance(other, JointState):
            return self.pstate == other.pstate and self.estate == other.estate
        return False
    def for_role(self, role):
        if role.lower().startswith("p"):
            return self.pstate
        else:
            return self.estate
    @classmethod
    def from_dict(self, states):
        return JointState(states["pursuer"], states["evader"])
    def copy(self):
        return JointState(self.pstate.copy(), self.estate.copy())
    def set_role(self, role, state):
        if role == "pursuer":
            self.pstate = state
        elif role == "evader":
            self.estate = state
        else:
            raise ValueError("Invalid role %s" % role)
    def __repr__(self):
        return "%s(P(%s),E(%s))" % (self.__class__.__name__, self.pstate, self.estate)

        
#-------
class GameAction(pomdp_py.Action):
    def __init__(self, role, name):
        self.role = role
        self.name = name
    def __hash__(self):
        return hash((self.name, self.role))
    def __eq__(self, other):
        if isinstance(other, GameAction):        
            return self.role == other.role and self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.name)

class PAction(GameAction):
    def __init__(self, name):
        super().__init__("pursuer", name)
    def copy(self):
        return PAction(self.name)

class EAction(GameAction):
    def __init__(self, name):
        super().__init__("evader", name)
    def copy(self):
        return EAction(self.name)    

class JointAction(pomdp_py.Action):
    def __init__(self, paction, eaction):
        self.paction = paction
        self.eaction = eaction
    def __hash__(self):
        return hash((self.paction, self.eaction))
    def __eq__(self, other):
        if isinstance(other, JointAction):
            return self.paction == other.paction and self.eaction == other.eaction
        return False
    def for_role(self, role):
        if role.lower().startswith("p"):
            return self.paction
        else:
            return self.eaction
    @classmethod
    def from_dict(self, actions):
        return JointAction(actions["pursuer"], actions["evader"])
    def copy(self):
        return JointAction(self.paction.copy(), self.eaction.copy())
    def set_role(self, role, action):
        if role == "pursuer":
            self.paction = action
        elif role == "evader":
            self.eaction = action
        else:
            raise ValueError("Invalid role %s" % role)
    def tolist(self):
        return [self.paction, self.eaction]
    def __repr__(self):
        return "%s(P(%s),E(%s))" % (self.__class__.__name__, self.paction, self.eaction)
        
#----
class Observation(pomdp_py.Observation):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Observation(%s)" % self.name

def other_side(side):
    if side == "left-door":
        return "right-door"
    else:
        return "left-door"

def other_role(side):
    if side == "pursuer":
        return "evader"
    else:
        return "pursuer" 
