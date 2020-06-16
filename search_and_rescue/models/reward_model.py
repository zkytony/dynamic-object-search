import pomdp_py
from search_and_rescue.env.action import *
from search_and_rescue.env.state import *

class DeterministicRewardModel(pomdp_py.RewardModel):
    def probability(self, reward, state, action, next_state, normalized=False, **kwargs):
        if reward == self._reward_func(state, action):
            return 1.0
        else:
            return 0.0
        
    def sample(self, state, action, next_state, normalized=False):
        # deterministic
        return self._reward_func(state, action, next_state)
    
    def argmax(self, state, action, next_state, normalized=False):
        """Returns the most likely reward"""
        return self._reward_func(state, action, next_state)


class GoalRewardModel(DeterministicRewardModel):
    """
    This is a reward where the agent gets reward only for detect-related actions.
    """
    def __init__(self, robot_id, target_objects, big=100, small=1):
        """
        robot_id (int): This model is the reward for one agent (i.e. robot),
                        If None, then this model could be for the environment.
        target_objects (set): a set of objids for target objects.
        """
        self._robot_id = robot_id
        self.big = big
        self.small = small
        self._target_objects = target_objects
            
    def _reward_func(self, state, action, next_state):
        reward = 0
        robot_id = self._robot_id

        # If the robot has detected all objects
        if len(state.object_states[robot_id]['objects_found'])\
           == len(self._target_objects):
            return 0  # no reward or penalty; the task is finished.

        if isinstance(action, MotionAction):
            reward = reward - self.small - action.distance_cost
        elif isinstance(action, LookAction):
            reward = reward - self.small
        elif isinstance(action, FindAction):
            if not state.object_states[robot_id]['camera_on']:
                # The robot didn't look before detect. So nothing is in the field of view.
                reward -= self.big
            else:
                # transition function should've taken care of the detection.
                new_objects_count = len(set(next_state.object_states[robot_id].objects_found)\
                                        - set(state.object_states[robot_id].objects_found))
                if new_objects_count == 0:
                    # No new detection. "detect" is a bad action.
                    reward -= self.big
                else:
                    # Has new detection. Award.
                    reward += self.big
        return reward
    
    
class AdversarialRewardModel(DeterministicRewardModel):
    """Adversarial agent reward model"""

    def __init__(self, object_id, robot_id, robot_sensor, big=100, small=1):
        """
        Object `object_id` is trying to act adversarially against the robot `robot_id`.
        The reward here is assigned to the adversarial object. 
        """
        self._object_id = object_id
        self._robot_id = robot_id
        self.big = big
        self.small = small
        self._robot_sensor = robot_sensor
        # self._keep_dist = keep_dist

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
        cur_in_range = self._robot_sensor.within_range(state.pose(self._robot_id),
                                                       state.pose(self._object_id))
        next_in_range = self._robot_sensor.within_range(next_state.pose(self._robot_id),
                                                        next_state.pose(self._object_id))
        print(cur_in_range, next_in_range)
        if not cur_in_range and next_in_range:
            return -self.big
        else:
            return -self.small


def unittest():
    from search_and_rescue.models.grid_map import GridMap
    from search_and_rescue.models.sensor import Laser2DSensor
    grid_map = GridMap(10, 10,
                       {0: (2,3), 5:(4,9)})
    motion_actions = create_motion_actions(can_stay=True)

    state = JointState({3: VictimState(3,   (4,8,math.pi), False),
                        4: SearcherState(4, (4,4,math.pi*2/3), (), False),
                        0: ObstacleState(0, (2,3)),
                        5: ObstacleState(5, (4,9))})
    next_state = JointState({3: VictimState(3,   (4,8,math.pi), False),
                             4: SearcherState(4, (4,5,math.pi*2/3), (3,), False),
                             0: ObstacleState(0, (2,3)),
                             5: ObstacleState(5, (4,9))})    

    goal_reward_model = GoalRewardModel(4, {3,5})
    print(goal_reward_model.sample(state, Find, next_state))
    
    state.object_states[4]["camera_on"] = True    
    next_state.object_states[4]["camera_on"] = True
    print(goal_reward_model.sample(state, Find, next_state))

    sensor = Laser2DSensor(4, fov=359, max_range=3, epsilon=0.8, sigma=0.2)    
    adversarial_reward_model = AdversarialRewardModel(3, 4, sensor)
    print(adversarial_reward_model.sample(state, MoveSouth, next_state))

if __name__ == '__main__':
    unittest()
        
