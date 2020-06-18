import pomdp_py

###### Observation ######
class ObjectObservation(pomdp_py.Observation):
    """The xy pose of the object is observed; or NULL if not observed"""
    NULL = None
    def __init__(self, objid, pose):
        self.objid = objid
        self.pose = pose
        
    def __hash__(self):
        return hash((self.objid, self.pose))
    
    def __eq__(self, other):
        if not isinstance(other, ObjectObservation):
            return False
        else:
            return self.objid == other.objid\
                and self.pose == other.pose

    def __str__(self):
        return ("ObjectObservation(%d, %s)" % (self.objid, str(self.pose)))

    def __repr__(self):
        return str(self)

    
class JointObservation(pomdp_py.OOObservation):
    """Full obserfvation"""
    def __init__(self, objposes):
        """
        objposes (dict): map from objid to 2d pose or NULL (not ObjectObservation!).
        """
        self._hashcode = hash(frozenset(objposes.items()))
        self.objposes = objposes

    def for_obj(self, objid):
        if objid in self.objposes:
            return ObjectObservation(objid, self.objposes[objid])
        else:
            return ObjectObservation(objid, ObjectObservation.NULL)
        
    def __hash__(self):
        return self._hashcode
    
    def __eq__(self, other):
        if not isinstance(other, JointObservation):
            return False
        else:
            return self.objposes == other.objposes

    def __str__(self):
        return "JointObservation(%s)" % str(self.objposes)

    def __repr__(self):
        return str(self)

    def factor(self, next_state, *params, **kwargs):
        """Factor this OO-observation by objects"""
        return {objid: ObjectObservation(objid, self.objposes[objid])
                for objid in next_state.object_states
                if objid != next_state.robot_id}
    
    @classmethod
    def merge(cls, object_observations, next_state, *params, **kwargs):
        """Merge `object_observations` into a single OOObservation object;
        
        object_observation (dict): Maps from objid to ObjectObservation"""
        return JointObservation({objid: object_observations[objid].pose
                                 for objid in object_observations
                                 if objid != next_state.object_states[objid].objclass != "robot"})
    

class ObservationCollection(pomdp_py.Observation):
    def __init__(self, observations):
        self.observations = observations

    def __getitem__(self, agent_id):
        if agent_id in self.observations:
            return self.observations[agent_id]
        else:
            return None

    def __str__(self):
        res = "ObservationCollection\n"
        for agent_id in self.observations:
            res += "  %d: %s\n" % (agent_id, str(self.observations[agent_id]))
        return res

    def __repr__(self):
        return "ObservationCollection(%s)" % str(self.observations)
    
