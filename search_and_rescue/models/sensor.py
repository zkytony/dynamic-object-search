#pass#pass
import pomdp_py
import random
import numpy as np
import math
from search_and_rescue.env.observation import *
from search_and_rescue.utils import *

class NoisySensor(pomdp_py.GenerativeDistribution):
    LASER = "laser"
    PROXIMITY = "proximity"
    def observe(self, agent_pose, env_state):
        """
        Returns an Observation with this sensor model.
        """
        raise NotImplementedError

    def within_range(self, agent_pose, point):
        """Returns true if the point is within range of the sensor; but the point might not
        actually be visible due to occlusion or "gap" between beams"""
        raise ValueError

    @property
    def sensing_region_size(self):
        return self._sensing_region_size

    def probability(self, observation, next_state, action, **kwargs):
        raise NotImplementedError
    
    def random(self, next_state, action, **kwargs):
        raise NotImplementedError
    
    def mpe(self, next_state, action, **kwargs):
        raise NotImplementedError
    

# Note that the occlusion of an object is implemented based on
# whether a beam will hit an obstacle or some other object before
# that object. Because the world is discretized, this leads to
# some strange pattern of the field of view. But what's for sure
# is that, when occlusion is enabled, the sensor will definitely
# not receive observation for some regions in the field of view
# making it a more challenging situation to deal with.

    
class Laser2DSensor(NoisySensor):
    """Fan shaped 2D laser sensor"""

    def __init__(self, agent_id,
                 fov=90, min_range=1, max_range=5,
                 angle_increment=5,
                 occlusion_enabled=False,
                 sigma=0, epsilon=1):
        """
        fov (float): angle between the start and end beams of one scan (degree).
        min_range (int or float)
        max_range (int or float)
        angle_increment (float): angular distance between measurements (rad).
        """
        self.agent_id = agent_id
        self.fov = to_rad(fov)  # convert to radian
        self.min_range = min_range
        self.max_range = max_range
        self.angle_increment = to_rad(angle_increment)
        self._occlusion_enabled = occlusion_enabled

        # determines the range of angles;
        # For example, the fov=pi, means the range scanner scans 180 degrees
        # in front of the agent. By our angle convention, 180 degrees maps to [0,90] and [270, 360]."""
        self._fov_left = (0, self.fov / 2)
        self._fov_right = (2*math.pi - self.fov/2, 2*math.pi)

        # beams that are actually within the fov (set of angles)
        self._beams = {round(th, 2)
                       for th in np.linspace(self._fov_left[0],
                                             self._fov_left[1],
                                             int(round((self._fov_left[1] - self._fov_left[0]) / self.angle_increment)))}\
                    | {round(th, 2)
                       for th in np.linspace(self._fov_right[0],
                                             self._fov_right[1],
                                             int(round((self._fov_right[1] - self._fov_right[0]) / self.angle_increment)))}
        # The size of the sensing region here is the area covered by the fan
        self._sensing_region_size = self.fov / (2*math.pi) * math.pi * (max_range - min_range)**2

        # noisy sensor parameters
        self.sigma = sigma
        self.epsilon = epsilon

    def in_field_of_view(th, view_angles):
        """Determines if the beame at angle `th` is in a field of view of size `view_angles`.
        For example, the view_angles=180, means the range scanner scans 180 degrees
        in front of the agent. By our angle convention, 180 degrees maps to [0,90] and [270, 360]."""
        fov_right = (0, view_angles / 2)
        fov_left = (2*math.pi - view_angles/2, 2*math.pi)
        
    def within_range(self, agent_pose, point):
        """Returns true if the point is within range of the sensor; but the point might not
        actually be visible due to occlusion or "gap" between beams"""
        dist, bearing = self.shoot_beam(agent_pose, point)
        if not in_range(dist, (self.min_range, self.max_range)):
            return False
        if (not in_range(bearing, self._fov_left))\
           and (not in_range(bearing, self._fov_right)):
            return False        
        return True

    def shoot_beam(self, agent_pose, point):
        """Shoots a beam from agent_pose at point. Returns the distance and bearing
        of the beame (i.e. the length and orientation of the beame)"""
        if len(agent_pose) == 2:
            rx, ry = agent_pose
            rth = 0
            print("HEY")
        else:
            rx, ry, rth = agent_pose
        dist = euclidean_dist(point, (rx,ry))
        bearing = (math.atan2(point[1] - ry, point[0] - rx) - rth) % (2*math.pi)  # bearing (i.e. orientation)
        return (dist, bearing)

    def valid_beam(self, dist, bearing):
        """Returns true beam length (i.e. `dist`) is within range and its angle
        `bearing` is valid, that is, it is within the fov range and in
        accordance with the angle increment."""
        return dist >= self.min_range and dist <= self.max_range\
            and round(bearing, 2) in self._beams

    def _build_beam_map(self, beam, point, beam_map={}):
        """beam_map (dict): Maps from bearing to (dist, point)"""
        dist, bearing = beam
        valid = self.valid_beam(dist, bearing)
        if not valid:
            return
        bearing_key = round(bearing,2)
        if bearing_key in beam_map:
            # There's an object covered by this beame already.
            # see if this beame is closer
            if dist < beam_map[bearing_key][0]:
                # point is closer; Update beam map
                beam_map[bearing_key] = (dist, point)
            else:
                # point is farther than current hit
                pass
        else:
            beam_map[bearing_key] = (dist, point)

    # def observe(self, agent_pose, env_state):
    #     """
    #     Returns a MosObservation with this sensor model.
    #     """
    #     rx, ry, rth = agent_pose
        
    #     # Check every object
    #     objposes = {}
    #     beam_map = {}
    #     for objid in env_state.object_states:
    #         objposes[objid] = ObjectObservation.NULL
    #         object_pose = env_state.object_states[objid]["pose"]
    #         beam = self.shoot_beam(agent_pose, object_pose)

    #         if not self._occlusion_enabled:
    #             if self.valid_beam(*beam):
    #                 d, bearing = beam  # distance, bearing
    #                 lx = rx + int(round(d * math.cos(rth + bearing)))
    #                 ly = ry + int(round(d * math.sin(rth + bearing)))                            
    #                 objposes[objid] = (lx, ly)
    #         else:
    #             self._build_beam_map(beam, object_pose, beam_map=beam_map)

    #     if self._occlusion_enabled:
    #         # The observed objects are in the beam_map
    #         for bearing_key in beam_map:
    #             d, objid = beam_map[bearing_key]
    #             lx = rx + int(round(d * math.cos(rth + bearing_key)))
    #             ly = ry + int(round(d * math.sin(rth + bearing_key)))
    #             objposes[objid] = (lx, ly)

    #     return MosOOObservation(objposes)

    @property
    def sensing_region_size(self):
        raise NotImplementedError

# class FovObservationModel(pomdp_py.ObservationModel):
#     """Field of view observation model;
#     Pasted from previous ObjectObservationModel"""
#     def __init__(self, objid, sensor, dim, sigma=0, epsilon=1, look_after_move=False):
#         """
#         sigma and epsilon are parameters of the observation model (see paper),
#         dim (tuple): a tuple (width, length) for the dimension of the world"""
#         self._objid = objid
#         self._sensor = sensor
#         self.sigma = sigma
#         self.epsilon = epsilon
#         self._look_after_move = look_after_move

    # Functions related to computing the probability and random sampling of observation.
    def _compute_params(self, object_in_sensing_region):
        if object_in_sensing_region:
            # Object is in the sensing region
            alpha = self.epsilon
            beta = (1.0 - self.epsilon) / 2.0
            gamma = (1.0 - self.epsilon) / 2.0
        else:
            # Object is not in the sensing region.
            alpha = (1.0 - self.epsilon) / 2.0
            beta = (1.0 - self.epsilon) / 2.0
            gamma = self.epsilon
        return alpha, beta, gamma
        
    def probability(self, observation, next_state, action, **kwargs):
        """
        Returns the probability of Pr (observation | next_state, action).

        Args:
            observation (ObjectObservation)
            next_state (State)
            action (Action)
        """
        if not self._is_sensing(action):
            # No observation should be received
            if observation.pose == ObjectObservation.NULL:
                return 1.0
            else:
                return 0.0

        # The (funny) business of allowing histogram belief update using O(oi|si',sr',a).
        next_agent_state = kwargs.get("next_agent_state", None)
        if next_agent_state is not None:
            assert next_agent_state["id"] == self._sensor.agent_id,\
                "Agent id of observation model mismatch with given state"
            agent_pose = next_agent_state.pose
            
            if isinstance(next_state, ObjectState):
                assert next_state["id"] == self.agent_id,\
                    "Object id of observation model mismatch with given state"
                object_pose = next_state.pose
            else:
                object_pose = next_state.pose(observation.objid)
        else:
            agent_pose = next_state.pose(self.agent_id)
            object_pose = next_state.pose(observation.objid)

        # Compute the probability
        zi = observation.pose
        alpha, beta, gamma = self._compute_params(self._sensor.within_range(agent_pose, object_pose))
        # Requires Python >= 3.6
        event_occured = random.choices(["A", "B", "C"], weights=[alpha, beta, gamma], k=1)[0]
        if event_occured == "A":
            # object in sensing region and observation comes from object i
            if zi == ObjectObservation.NULL:
                # Even though event A occurred, the observation is NULL.
                # This has 0.0 probability.
                return 0.0 * alpha
            else:
                gaussian = pomdp_py.Gaussian(list(object_pose),
                                             [[self.sigma**2, 0],
                                              [0, self.sigma**2]])
                return gaussian[zi] * alpha
        elif event_occured == "B":
            return (1.0 / self._sensor.sensing_region_size) * beta

        else: # event_occured == "C":
            prob = 1.0 if zi == ObjectObservation.NULL else 0.0  # indicator zi == NULL
            return prob * gamma
            

    def random(self, next_state, action, object_id=None, grid_map=None, argmax=False):
        """Returns observation"""
        assert object_id is not None, "Did not specify which object to create observation for"
        agent_pose = next_state.pose(self.agent_id)
        object_pose = next_state.pose(object_id)

        # Obtain observation according to distribution. 
        alpha, beta, gamma = self._compute_params(self.within_range(agent_pose, object_pose))

        # Requires Python >= 3.6
        if not argmax:
            event_occured = random.choices(["A", "B", "C"], weights=[alpha, beta, gamma], k=1)[0]
        else:
            event_probs = {"A": alpha,
                           "B": beta,
                           "C": gamma}
            event_occured = max(event_probs, key=lambda e: event_probs[e])
        zi = self._sample_zi(event_occured, next_state,
                             grid_map, argmax=argmax)

        return ObjectObservation(self.agent_id, zi)

    def mpe(self, next_state, action, object_id=None, grid_map=None):
        return self.random(next_state, action,
                           object_id=object_id, grid_map=grid_map, argmax=True)

    def _sample_zi(self, event, next_state, grid_map, argmax=False):
        if event == "A":
            object_true_pose = next_state.pose(self.agent_id)
            gaussian =  pomdp_py.Gaussian(list(object_true_pose[:2]),
                                          [[self.sigma**2, 0],
                                           [0, self.sigma**2]])
            if not argmax:
                zi = gaussian.random()
            else:
                zi = gaussian.mpe()
            zi = (int(round(zi[0])), int(round(zi[1])))
                
        elif event == "B":
            zi = random.sample(grid_map.free_poses, 1)[0]
        else: # event == C
            zi = ObjectObservation.NULL
        return zi
