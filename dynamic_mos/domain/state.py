"""Defines the State for the 2D Multi-Object Search domain;

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of
topological graph)

Description: Multi-Object Search in a 2D grid world.

State space: 

    :math:`S_1 \\times S_2 \\times ... S_n \\times S_r`
    where :math:`S_i (1\leq i\leq n)` is the object state, with attribute
    "pose" :math:`(x,y)` and Sr is the state of the robot, with attribute
    "pose" :math:`(x,y)` and "objects_found" (set).
"""

import pomdp_py
import math

###### States ######
class ObjectState(pomdp_py.ObjectState):
    def __init__(self, objid, objclass, pose, time=-1):
        """If the object is static, then `time` is -1. Otherwise, `time`
        indicates the time step of the dynamic object."""
        if objclass != "obstacle" and objclass != "target":
            raise ValueError("Only allow object class to be"\
                             "either 'target' or 'obstacle'."
                             "Got %s" % objclass)
        super().__init__(objclass, {"pose":pose,
                                    "id":objid,
                                    "time":time})
    def __str__(self):
        return 'ObjectState(%s,%s,t=%d)' % (str(self.objclass), str(self.pose), self.time)
    @property
    def pose(self):
        return self.attributes['pose']
    @property
    def objid(self):
        return self.attributes['id']
    @property
    def time(self):
        return self.attributes['time']

class RobotState(pomdp_py.ObjectState):
    def __init__(self, robot_id, pose, objects_found, camera_direction):
        """Note: camera_direction is None unless the robot is looking at a direction,
        in which case camera_direction is the string e.g. look+x, or 'look'"""
        super().__init__("robot", {"id":robot_id,
                                   "pose":pose,  # x,y,th
                                   "objects_found": objects_found,
                                   "camera_direction": camera_direction})
    def __str__(self):
        return 'RobotState(%s,%s|%s)' % (str(self.objclass), str(self.pose), str(self.objects_found))
    def __repr__(self):
        return str(self)
    @property
    def pose(self):
        return self.attributes['pose']
    @property
    def robot_pose(self):
        return self.attributes['pose']
    @property
    def objects_found(self):
        return self.attributes['objects_found']

class MosOOState(pomdp_py.OOState):
    def __init__(self, robot_id, object_states):
        self.robot_id = robot_id
        super().__init__(object_states)
    def object_pose(self, objid):
        return self.object_states[objid]["pose"]
    def pose(self, objid):
        return self.object_pose(objid)
    @property
    def object_poses(self):
        return {objid:self.object_states[objid]['pose']
                for objid in self.object_states}
    @property
    def robot_state(self):
        return self.object_states[self.robot_id]
    @property
    def robot_pose(self):
        return self.object_states[self.robot_id]["pose"]    
    def __str__(self):
        return 'MosOOState%s' % (str(self.object_states))
    def __repr__(self):
        return str(self)
