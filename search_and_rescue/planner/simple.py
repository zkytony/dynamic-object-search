import pomdp_py
from search_and_rescue.planner.greedy import *
from search_and_rescue.env.action import *

class SimpleReactivePlanner(ManualPlanner):

    def __init__(self, role, motion_policy, look_after_move=True):
        if not look_after_move:
            raise ValueError("Simple Reactive Planner won't take"
                             "information-gathering action. So only"
                             "works when there is NO LOOK action.")
        self._motion_policy = motion_policy
        self._role = role
        super().__init__(look_after_move=look_after_move)

    def plan(self, agent):
        if self._find_next:
            self._find_next = False
            return FindAction()
        
        state = agent.belief.mpe()
        action = self._motion_policy.random(state, get_action=True)
        return action

    def update(self, agent, real_action, real_observation, **kwargs):
        if self._role == "searcher":
            super().update(agent, real_action, real_observation, **kwargs)
