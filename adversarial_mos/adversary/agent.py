import pomdp_py
import random
import copy
from dynamic_mos.models.components.motion_policy import StochaisticPolicy, next_pose, AdversarialPolicy
from dynamic_mos.models.dynamic_transition_model import *
from dynamic_mos.models.transition_model import RobotTransitionModel
from dynamic_mos.models.observation_model import ObjectObservationModel
from dynamic_mos.domain.observation import *
from dynamic_mos.domain.state import *
from dynamic_mos.utils import *
from dynamic_mos.agent.belief import MosOOBelief

class AdversarialRewardModel(pomdp_py.RewardModel):
    """Adversarial agent reward model"""
    def __init__(self, object_id, robot_id, keep_dist=2, big=100, small=1):
        self._robot_id = robot_id
        self._object_id = object_id
        self.big = big
        self.small = small
        self._keep_dist = keep_dist

    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        if reward == self._reward_func(state, action):
            return 1.0
        else:
            return 0.0
        
    def sample(self, state, action, next_state,
               normalized=False, robot_id=None):
        # deterministic
        return self._reward_func(state, action, next_state)
    
    def argmax(self, state, action, next_state, normalized=False):
        """Returns the most likely reward"""
        return self._reward_func(state, action, next_state)
    
    def _reward_func(self, state, action, next_state):
        # if the adversarial has been detected by the robot, then -big.
        # otherwise, -small
        reward = 0
        next_dist = euclidean_dist(state.pose(self._object_id),
                                   state.pose(self._robot_id))
        if next_dist < self._keep_dist:
            reward -= self.big
        else:
            reward -= self.small
        return reward

    
class BasicMotionPolicy(StochaisticPolicy):
    """Adversarial Target motion policy."""
    def __init__(self, object_id, grid_map, motion_actions):
        self._object_id = object_id
        self._motion_actions = motion_actions
        super().__init__(grid_map, motion_actions)

    def probability(self, next_object_state, state, action):
        if next_object_state["id"] != self._object_id:
            return 1e-9
        
        legal_actions = self._legal_actions[cur_object_state.pose]
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
        if action not in self._legal_actions[state.object_pose(self._object_id)]:
            # raise ValueError("Action %s cannot be taken in state %s" % (str(action), str(state)))
            # Action is not lega. Does not move.
            return next_object_state

        cur_object_pose = state.pose(self._object_id)
        next_object_pose = next_pose(cur_object_pose, action.motion)
        next_object_state["pose"] = next_object_pose
        return next_object_state

class AdversarialObservationModel(pomdp_py.ObservationModel):
    def __init__(self, object_id, robot_id):
        self._object_id = object_id
        self._robot_id = robot_id
    
    def probability(self, observation, next_state, action, **kwargs):
        """
        probability(self, observation, next_state, action, **kwargs)
        Returns the probability of :math:`\Pr(o|s',a)`.

        Args:
            observation (~pomdp_py.framework.basics.Observation): the observation :math:`o`
            next_state (~pomdp_py.framework.basics.State): the next state :math:`s'`
            action (~pomdp_py.framework.basics.Action): the action :math:`a`
        Returns:
            float: the probability :math:`\Pr(o|s',a)`
        """
        assert isinstance(observation, ObjectObservation)
        if observation.objid != self._robot_id:
            return 1e-9
        if observation.pose != next_state.pose(self._robot_id):
            return 1e-9
        else:
            return 1.0 - 1e-9
    
    def sample(self, next_state, action, **kwargs):
        """sample(self, next_state, action, **kwargs)
        Returns observation randomly sampled according to the
        distribution of this observation model.

        Args:
            next_state (~pomdp_py.framework.basics.State): the next state :math:`s'`
            action (~pomdp_py.framework.basics.Action): the action :math:`a`
        Returns:
            Observation: the observation :math:`o`
        """
        return ObjectObservation(self._object_id, next_state.pose(self._robot_id))
        
    
    def argmax(self, next_state, action, **kwargs):
        """
        argmax(self, next_state, action, **kwargs)
        Returns the most likely observation"""
        return self.sample(next_state, action **kwargs)


    
class AdversarialTransitionModel(pomdp_py.OOTransitionModel):
    def __init__(self, object_id, robot_id, grid_map, motion_policy):
        """
        Args:
            motion_actions (list) Motion actions of the adversarial object
            robot_sensor (Sensor): sensor of the robot.
        """
        assert motion_policy._object_id == object_id
        self.motion_policy = motion_policy
        # From the adversarial object's perspective, the robot is a greedy
        # agent, and the object itself has the dynamic agent transition model
        # which simply moves the object around.
        transition_models = {
            object_id: DynamicAgentTransitionModel(object_id, self.motion_policy),
            robot_id: DynamicObjectTransitionModel(
                robot_id,
                AdversarialPolicy(grid_map, -1,
                                  pr_stay=1e-9,
                                  rule="chase",
                                  motion_actions=motion_policy.motion_actions),
                robot_id=object_id)}  # treating the object itself as the 'intelligent agent'
        self._object_id = object_id
        super().__init__(transition_models)

    def sample(self, state, action, **kwargs):
        oostate = pomdp_py.OOTransitionModel.sample(self, state, action, **kwargs)
        return MosOOState(state.robot_id, oostate.object_states)

    def argmax(self, state, action, normalized=False, **kwargs):
        oostate = pomdp_py.OOTransitionModel.argmax(self, state, action, **kwargs)
        return MosOOState(state.robot_id, oostate.object_states)


class AdversarialPolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, object_id, motion_policy, action_prior=None):
        self.motion_policy = motion_policy
        self.object_id = object_id
        self.action_prior = action_prior
        
    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]

    def get_all_actions(self, state=None, history=None):
        if state is None:
            return self.motion_policy.motion_actions
        else:
            return self.motion_policy.legal_actions[state.pose(self.object_id)]
                
    def rollout(self, state, history):
        if self.action_prior is None:
            return random.sample(self.get_all_actions(state=state, history=history), 1)[0]
        else:
            preferences = self.action_prior.get_preferred_actions(state, history)
            if len(preferences) > 0:
                return random.sample(preferences, 1)[0][0]
            else:
                return random.sample(self.get_all_actions(state=state, history=history), 1)[0]


class AdversarialActionPrior(pomdp_py.ActionPrior):
    def __init__(self, object_id, robot_id, grid_map,
                 num_visits_init, val_init, motion_policy):
        self.object_id = object_id
        self.robot_id = robot_id
        self.grid_map = grid_map
        self.motion_policy = motion_policy
        self.num_visits_init = num_visits_init
        self.val_init = val_init

    def get_preferred_actions(self, state, history):
        # Prefer actions that move the target away from the robot.
        cur_dist = euclidean_dist(state.pose(self.object_id),
                                  state.pose(self.robot_id))
        preferences = set()
        for action in self.motion_policy.legal_actions[state.pose(self.object_id)]:
            next_object_pose = next_pose(state.pose(self.object_id), action.motion)
            if next_object_pose not in self.grid_map.obstacle_poses:
                next_dist = euclidean_dist(next_object_pose,
                                           state.pose(self.robot_id))
                if next_dist > cur_dist:
                    preferences.add((action, self.num_visits_init, self.val_init))
        return preferences


class AdversarialTarget(pomdp_py.Agent):
    def __init__(self,
                 object_id,
                 init_object_state,
                 object_sensor,
                 motion_policy,  # motion actions of the target
                 grid_map,
                 robot_id,
                 **kwargs):
        """
        kwargs include:
        
            sigma, epsilon (observation model parameters)
            belief_rep="histogram" (belief representation)
            prior={},  # does the target know where the robot is?
            grid_map,
            big=100,
            small=1,
            action_prior=None
        """
        self._object_id = object_id
        self._object_sensor = object_sensor
        self._grid_map = grid_map
        self._robot_id = robot_id

        transition_model = AdversarialTransitionModel(object_id, robot_id, grid_map, motion_policy)

        # The observation model observes the roboot
        observation_model = AdversarialObservationModel(object_id, robot_id)
        # observation_model = ObjectObservationModel(robot_id, object_sensor,
        #                                            (self._grid_map.width, self._grid_map.length),
        #                                            sigma=kwargs.get("sigma", 0),
        #                                            epsilon=kwargs.get("epsilon", 1),
        #                                            look_after_move=True)

        reward_model = AdversarialRewardModel(object_id, robot_id,
                                              big=kwargs.get("big", 100),
                                              small=kwargs.get("small", 1))
        
        policy_model = AdversarialPolicyModel(object_id, transition_model.motion_policy,
                                              action_prior=kwargs.get("action_prior", None))

        init_belief = self._init_hist_belief(kwargs.get("prior", {}),
                                             init_object_state)
        super().__init__(init_belief, policy_model,
                         transition_model=transition_model,
                         observation_model=observation_model,
                         reward_model=reward_model)


    def _init_hist_belief(self, prior, init_object_state):
        """Initialize histogram belief"""
        if self.cur_belief is not None:
            raise ValueError("Agent already has initial belief!")
        
        oo_hists = {self._object_id: pomdp_py.Histogram({init_object_state:1.0})}

        hist = {}        
        if self._robot_id in prior:
            for pose in prior[self._robot_id]:
                state = ObjectState(self._robot_id,
                                    "robot",
                                    pose,
                                    time=0)
                hist[state]= prior[self._robot_id][pose]
        else:
            for x in range(self._grid_map.width):
                for y in range(self._grid_map.length):
                    if (x,y) in self._grid_map.obstacle_poses:
                        continue
                    state = ObjectState(self._robot_id,
                                        "robot",
                                        pose,
                                        time=0)
                    hist[state] = 1.0
                    total_prob += hist[state]
            # Normalize
            for state in hist:
                hist[state] /= total_prob

        hist_belief = pomdp_py.Histogram(hist)
        oo_hists[self._robot_id] = hist_belief
        return MosOOBelief(self._robot_id, oo_hists)
            
