from dynamic_mos.models.reward_model import *


class AdversarialRewardModel(pomdp_py.RewardModel):

    def __init__(self, object_id, robot_id, big=100, small=1):
        self._robot_id = robot_id
        self._object_id = object_id
        self.big = big
        self.small = small

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
        if self._object_id in next_state.object_states[self._robot_id]['objects_found']:
            reward -= self.big
        else:
            reward -= self.small
        return reward
