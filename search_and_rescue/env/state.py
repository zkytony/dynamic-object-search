import pomdp_py

class ObjectState(pomdp_py.ObjectState):
    @property
    def pose(self):
        return self.attributes['pose']
    @property
    def id(self):
        return self.attributes['id']
    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, str(self.attributes))
    def __repr__(self):
        return str(self)    

class ObstacleState(ObjectState):
    def __init__(self, objid, pose):
        super().__init__("obstacle", {"id":objid,
                                      "pose":pose})

class TargetState(ObjectState):
    """Static target"""
    def __init__(self, objid, pose):
        super().__init__("target", {"id":objid,
                                      "pose":pose})        

class DynamicObjectState(ObjectState):
    @property
    def time(self):
        return self["time"]

class SearcherState(DynamicObjectState):
    def __init__(self, objid, pose, objects_found, camera_on, time=-1):
        super().__init__("searcher", {"id":objid,
                                      "pose":pose,  # x,y,th
                                      "objects_found": objects_found,
                                      "camera_on": camera_on,
                                      "time": time})
    @property
    def objects_found(self):
        return self["objects_found"]


class VictimState(DynamicObjectState):
    def __init__(self, objid, pose, fov_objects, camera_on, time=-1):
        super().__init__("victim", {"id":objid,
                                    "pose":pose,  # x,y,th
                                    "camera_on": camera_on,
                                    "fov_objects": fov_objects, # objects in field of view
                                    "time": time})
    @property
    def fov_objects(self):
        return self["fov_objects"]        

class SuspectState(DynamicObjectState):
    def __init__(self, objid, pose, fov_objects, camera_on, time=-1):
        super().__init__("suspect", {"id":objid,
                                     "pose":pose,  # x,y,th
                                     "camera_on": camera_on,
                                     "fov_objects": fov_objects,
                                     "time": time})
    @property
    def fov_objects(self):
        return self["fov_objects"]        

class JointState(pomdp_py.OOState):
    def __init__(self, object_states):
        super().__init__(object_states)
    def pose(self, objid):
        return self.object_states[objid]["pose"]
    def __str__(self):
        return 'JointState(%s)' % (str(self.object_states))
    def __repr__(self):
        return str(self)    
