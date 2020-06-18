import pomdp_py
from search_and_rescue.env.action import *
from search_and_rescue.env.state import *

class SARRewardModel(pomdp_py.RewardModel):
    def __init__(self, agent_id, big=100, small=1, role_to_ids={}):
        print(small)
        self._agent_id = agent_id
        self.big = big
        self.small = small
        if "suspect" not in role_to_ids:
            role_to_ids["suspect"] = set()
        if "victim" not in role_to_ids:
            role_to_ids["victim"] = set()
        if "searcher" not in role_to_ids:
            role_to_ids["searcher"] = set()
        if "target" not in role_to_ids:
            role_to_ids["target"] = set()
        self._role_to_ids = role_to_ids

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action, next_state)

class SearcherRewardModel(SARRewardModel):
    """
    This is a reward where the agent gets reward only for detect-related actions.
    """
    def __init__(self, agent_id, big=100, small=1, role_to_ids={}):
        super().__init__(agent_id, big=big, small=small, role_to_ids=role_to_ids)
        self._target_ids = self._role_to_ids["suspect"]\
            | self._role_to_ids["victim"] | self._role_to_ids["target"]
        
    def _reward_func(self, state, action, next_state):
        reward = 0
        # Detected all objects
        if self._target_ids.issubset(state.object_states[self._agent_id]['objects_found']):
            return 0  # no reward or penalty; the task is finished.

        if isinstance(action, MotionAction):
            reward = reward - self.small - action.distance_cost

        elif isinstance(action, LookAction):
            reward = reward - self.small
            
        elif isinstance(action, FindAction):
            if not state.object_states[self._agent_id]['camera_on']:
                # The robot didn't look before detect. So nothing is in the field of view.
                reward -= self.big
            else:
                # transition function should've taken care of the detection.
                new_objects_count = len(set(next_state.object_states[self._agent_id].objects_found)\
                                        - set(state.object_states[self._agent_id].objects_found))
                if new_objects_count == 0:
                    # No new detection. "detect" is a bad action.
                    reward -= self.big
                else:
                    # Has new detection. Award.
                    reward += self.big
        return reward

    
class SuspectRewardModel(SARRewardModel):
    """Adversarial agent reward model"""
    def __init__(self, agent_id, searcher_sensors, big=100, small=1, role_to_ids={}):
        self._searcher_sensors = searcher_sensors
        super().__init__(agent_id, big=big, small=small, role_to_ids=role_to_ids)

    def _reward_func(self, state, action, next_state):
        # If searcher can observe the suspect, then bad.
        # If suspect can observe victim then good.
        for searcher_id in self._role_to_ids["searcher"]:
            if searcher_id in self._searcher_sensors:
                sensor = self._searcher_sensors[searcher_id]
                cur_in_range = sensor.within_range(state.pose(searcher_id),
                                                   state.pose(self._agent_id))
                next_in_range = sensor.within_range(next_state.pose(searcher_id),
                                                    next_state.pose(self._agent_id))
                if not cur_in_range and next_in_range:
                    return -self.big
                else:
                    return -self.small
                
        # for victim_id in self._role_to_ids["victim"]:
        #     cur_seen_victim = victim_id in state.object_states[self._agent_id]["fov_objects"]
        #     next_seen_victim = victim_id in next_state.object_states[self._agent_id]["fov_objects"]
        #     if not cur_seen_victim and next_seen_victim:
        #         return self.big
        # print("HAHA")
        # return -self.small
    
    
class VictimRewardModel(SARRewardModel):
    """Adversarial agent reward model"""

    def _reward_func(self, state, action, next_state):
        # (NOT USED) If fov objects contains searcher, then good.
        # If fov objects contains suspect, then bad.
        # Otherwise, -small
        # for searcher_id in self._role_to_ids["searcher"]:
        #     cur_seen_searcher = searcher_id in state.object_states[self._agent_id]["fov_objects"]
        #     next_seen_searcher = searcher_id in next_state.object_states[self._agent_id]["fov_objects"]            
            
        #     if not cur_seen_searcher and next_seen_searcher:
        #         return self.big
        #     elif cur_seen_searcher and next_seen_searcher:
        #         return -self.small
        # for suspect_id in self._role_to_ids["suspect"]:
        #     cur_seen_suspect = suspect_id in state.object_states[self._agent_id]["fov_objects"]
        #     next_seen_suspect = suspect_id in next_state.object_states[self._agent_id]["fov_objects"]
        #     if not cur_seen_suspect and next_seen_suspect:
        #         return -self.big
        #     elif cur_seen_suspect and next_seen_suspect:
        #         return -self.small            
        # return -self.small
        return -self.small


def unittest():
    from search_and_rescue.models.motion_policy import BasicMotionPolicy, AdversarialPolicy
    from search_and_rescue.models.grid_map import GridMap
    from search_and_rescue.models.sensor import Laser2DSensor
    from search_and_rescue.models.transition_model import JointTransitionModel
    grid_map = GridMap(10, 10,
                       {0: (2,3), 5:(4,9)})
    motion_actions = create_motion_actions(can_stay=True)

    sensor3 = Laser2DSensor(3, fov=359, max_range=10, epsilon=0.8, sigma=0.2)        
    t3 = JointTransitionModel(3, sensor3,
                             {0,5}, {3,4,6},
                             {3: BasicMotionPolicy(3, grid_map, motion_actions),
                              4: AdversarialPolicy(4, 6, grid_map, 3, motion_actions=motion_actions, rule="chase"),
                              6: AdversarialPolicy(6, 3, grid_map, 3, motion_actions=motion_actions)})

    sensor4 = Laser2DSensor(4, fov=359, max_range=10, epsilon=0.8, sigma=0.2)        
    t4 = JointTransitionModel(4, sensor4,
                              {0,5}, {3,4,6},
                              {4: BasicMotionPolicy(4, grid_map, motion_actions),
                               3: AdversarialPolicy(3, 6, grid_map, 3, motion_actions=motion_actions),
                               6: AdversarialPolicy(6, 4, grid_map, 3, motion_actions=motion_actions)})

    sensor6 = Laser2DSensor(6, fov=359, max_range=10, epsilon=0.8, sigma=0.2)        
    t6 = JointTransitionModel(6, sensor6,
                              {0,5}, {3,4,6},
                              {6: BasicMotionPolicy(6, grid_map, motion_actions),
                               3: AdversarialPolicy(3, 4, grid_map, 3, motion_actions=motion_actions),
                               4: AdversarialPolicy(4, 6, grid_map, 3, motion_actions=motion_actions, rule="chase")})

    state = JointState({3: VictimState(3,   (4,8,math.pi), (), False),
                        4: SearcherState(4, (4,4,math.pi*2/3), (), False),
                        6: SuspectState(6, (2,2,math.pi*2/3), (), False),                        
                        0: ObstacleState(0, (2,3)),
                        5: ObstacleState(5, (4,9))})
    
    role_to_ids = {"searcher": {4},
                   "victim": {3},
                   "suspect": {6}}
    
    searcher_reward = SearcherRewardModel(4, role_to_ids=role_to_ids)
    victim_reward = VictimRewardModel(3, role_to_ids=role_to_ids)
    suspect_reward = SuspectRewardModel(6, {4:sensor4}, role_to_ids=role_to_ids)

    print(searcher_reward.sample(state, Look, t4.sample(state, Look)))
    print(victim_reward.sample(state, Look, t3.sample(state, Look)))
    print(suspect_reward.sample(state, Look, t6.sample(state, Look)))
    print(suspect_reward.sample(state, MoveSouth, t6.sample(state, MoveSouth)))        
    print(suspect_reward.sample(state, MoveNorth, t6.sample(state, MoveNorth)))    

if __name__ == '__main__':
    unittest()
