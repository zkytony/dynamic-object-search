from pursuit_evasion.domain import *
from pursuit_evasion.models import *
import pomdp_py


class MultiAgentRewardModel(pomdp_py.RewardModel):
    def __init__(self, reward_models):
        """role -> RewardModel"""
        self._reward_models = reward_models

    def __contains__(self, role):
        return role in self._reward_models

    def sample(self, state, action, next_state, role=None):
        return self._reward_models[role].sample(state, action, next_state)


    
class PursuitEvasionEnvironment(pomdp_py.Environment):

    def __init__(self, init_state):
        transition_model = TransitionModel()
        pursuer_reward = RewardModel("pursuer")
        evader_reward = RewardModel("evader")        
        reward_model = MultiAgentRewardModel({"pursuer":pursuer_reward,
                                              "evader":evader_reward})
        self._last_real_action = None
        super().__init__(init_state,
                         transition_model,
                         reward_model)

    def reset(self):
        self._last_real_action = None

    def __str__(self):
        string = ""
        
        sp = self.state.pstate
        se = self.state.estate
        sp_at_left_door = False
        if se.side == "left-door":
            sp_at_left_door = True
        else:
            sp_at_right_door = False
            
        se_at_left_door = False            
        if se.side == "left-door":
            se_at_left_door = True
        else:
            se_at_right_door = False

        left_door_string = "="
        right_door_string = "="
        if self._last_real_action is not None:
            if self._last_real_action.paction.name == "open-door":
                if sp_at_left_door:
                    left_door_string = "|"
                else:
                    right_string = "|"
            if self._last_real_action.eaction.name == "open-door":
                if se_at_left_door:
                    left_door_string = "|"
                else:
                    right_string = "|"

        if se_at_left_door:
            string += "E .\n"
        else:
            string += ". E\n"
        string += "%s|%s\n" % (left_door_string, right_door_string)

        if sp_at_left_door:
            string += "P ."
        else:
            string += ". P"
        return string


if __name__ == "__main__":
    pstate = PState("left-door")
    estate = EState("right-door")
    state = JointState(pstate, estate)
    env = PursuitEvasionEnvironment(state)
    print(env)