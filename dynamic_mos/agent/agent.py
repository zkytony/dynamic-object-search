# Defines the agent. There's nothing special
# about the MOS agent in fact, except that
# it uses models defined in ..models, and
# makes use of the belief initialization
# functions in belief.py
import pomdp_py
from .belief import *
from ..models.transition_model import *
from ..models.dynamic_transition_model import *
from ..models.observation_model import *
from ..models.reward_model import *
from ..models.policy_model import *

class MosAgent(pomdp_py.Agent):
    """One agent is one robot."""
    def __init__(self,
                 robot_id,
                 init_robot_state,  # initial robot state (assuming robot state is observable perfectly)
                 object_ids,  # target object ids
                 dim,         # tuple (w,l) of the width (w) and length (l) of the gridworld search space.
                 sensor,      # Sensor equipped on the robot
                 sigma=0.01,     # parameter for observation model
                 epsilon=1,   # parameter for observation model
                 belief_rep="histogram",  # belief representation, either "histogram" or "particles".
                 prior={},       # prior belief, as defined in belief.py:initialize_belief
                 num_particles=100,  # used if the belief representation is particles
                 grid_map=None,  # GridMap used to avoid collision with obstacles (None if not provided)
                 motion_policies={},  # map from dynamic object id to MotionPolicy
                 big=100,
                 small=1,
                 action_prior=None,
                 look_after_move=False):
        self.robot_id = robot_id
        self._object_ids = object_ids
        self.sensor = sensor
        self._dim = dim

        # since the robot observes its own pose perfectly, it will have 100% prior
        # on this pose.
        prior[robot_id] = {init_robot_state.pose: 1.0}
        rth = init_robot_state.pose[2]

        if len(motion_policies) > 0:
            static_object_ids = self._object_ids - set(motion_policies.keys())
            transition_model = DynamicMosTransitionModel(self._dim,
                                                         {self.robot_id: self.sensor},
                                                         static_object_ids,
                                                         motion_policies,
                                                         look_after_move=look_after_move)
        else:
            transition_model = MosTransitionModel(self._dim,
                                                  {self.robot_id: self.sensor},
                                                  self._object_ids,
                                                  look_after_move=look_after_move)
        observation_model = MosObservationModel(self._dim,
                                                self.sensor,
                                                self._object_ids,
                                                sigma=sigma,
                                                epsilon=epsilon,
                                                look_after_move=look_after_move)
        reward_model = GoalRewardModel(self._object_ids, robot_id=self.robot_id,
                                       big=big, small=small)
        if action_prior is None:
            policy_model = PolicyModel(self.robot_id, grid_map=grid_map,
                                       look_after_move=look_after_move)
        else:
            policy_model = PreferredPolicyModel(action_prior,
                                                look_after_move=look_after_move)

        super().__init__(None, policy_model,
                         transition_model=transition_model,
                         observation_model=observation_model,
                         reward_model=reward_model)
        # initialize belief        
        init_belief = initialize_belief(self,
                                        self._dim,
                                        self.robot_id,
                                        self._object_ids,
                                        prior=prior,
                                        representation=belief_rep,
                                        robot_orientations={self.robot_id:rth},
                                        num_particles=num_particles,
                                        grid_map=grid_map)
        self.set_belief(init_belief, prior=True)
        

    def clear_history(self):
        """Custum function; clear history"""
        self._history = None
        
    def motion_policy(self, objid):
        if isinstance(self.transition_model, DynamicMosTransitionModel):
            return self.transition_model.motion_policy(objid)
        return None
