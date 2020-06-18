import pomdp_py
from search_and_rescue.env.action import *
from search_and_rescue.env.state import *
from search_and_rescue.models.motion_policy import *
from search_and_rescue.models.transition_model import *
from search_and_rescue.models.observation_model import *
from search_and_rescue.models.policy_model import *
from search_and_rescue.models.reward_model import *
from search_and_rescue.models.sensor import *

class SARAgent(pomdp_py.Agent):

    def __init__(self,
                 agent_id,
                 init_belief,
                 policy_model,
                 transition_model,
                 observation_model,
                 reward_model):
        self.agent_id = agent_id
        super().__init__(init_belief,
                         policy_model,
                         transition_model=transition_model,
                         observation_model=observation_model,
                         reward_model=reward_model)

    @classmethod
    def construct(self, agent_id, agent_role, sensor, role_to_ids, grid_map, motion_actions,
                  look_after_move=False, num_visits_init=10, big=100, small=10,
                  prior={}, **kwargs):
        """
        prior (dict) mapping from object id to a dictionary (x,y,th) or (x,y) -> probability
        """

        # Construct motion policies
        motion_policies = {}
        for role in role_to_ids:
            for objid in role_to_ids[role]:
                if objid == agent_id:
                    mpoli = BasicMotionPolicy(agent_id, grid_map, motion_actions)
                else:
                    if role == "searcher":
                        # The searcher is chasing everybody
                        policies = []
                        for id2 in role_to_ids["victim"]:
                            p = AdversarialPolicy(
                                id2, objid, grid_map, None,  # None: unknown sensor range
                                motion_actions=motion_actions, rule="chase")
                            policies.append(p)
                        for id2 in role_to_ids["suspect"]:
                            p = AdversarialPolicy(
                                id2, objid, grid_map, None,  # None: unknown sensor range
                                motion_actions=motion_actions, rule="avoid")
                            policies.append(p)
                        mpoli = MixedPolicy(objid, policies)
                    elif role == "victim":
                        # victim is just trying to avoid the suspect
                        policies = []
                        for id2 in role_to_ids["suspect"]:
                            p = AdversarialPolicy(
                                id2, objid, grid_map, None,  # None: unknown sensor range
                                motion_actions=motion_actions, rule="chase")
                            policies.append(p)
                        mpoli = MixedPolicy(objid, policies)
                    elif role == "suspect":
                        # suspect tries to avoid searcher but chases the victim
                        policies = []
                        for id2 in role_to_ids["victim"]:
                            p = AdversarialPolicy(
                                id2, objid, grid_map, None,  # None: unknown sensor range
                                motion_actions=motion_actions, rule="avoid")
                            policies.append(p)
                        for id2 in role_to_ids["searcher"]:
                            p = AdversarialPolicy(
                                id2, objid, grid_map, None,  # None: unknown sensor range
                                motion_actions=motion_actions, rule="chase")
                            policies.append(p)
                        mpoli = MixedPolicy(objid, policies)
                motion_policies[objid] = mpoli
                
        # Construct transition model
        static_object_ids = role_to_ids["target"]
        dynamic_object_ids = role_to_ids["searcher"] | role_to_ids["victim"] | role_to_ids["suspect"]

        transition_model = JointTransitionModel(agent_id, sensor,
                                                static_object_ids, dynamic_object_ids,
                                                motion_policies, look_after_move=look_after_move)

        # Construct observation model
        observation_model = SensorModel(static_object_ids | dynamic_object_ids,
                                        sensor, grid_map, look_after_move=look_after_move)

        # Construct policy model
        if agent_role == "searcher":
            rules = {id2:"chase" for id2 in (role_to_ids["victim"] | role_to_ids["suspect"] | role_to_ids["target"])}
        elif agent_role == "victim":
            rules = {id2:"avoid" for id2 in (role_to_ids["suspect"])}
        elif agent_role == "suspect":
            rules_victim = {id2:"chase" for id2 in role_to_ids["victim"]}
            rules_searcher = {id2:"avoid" for id2 in role_to_ids["searcher"]}
            rules = {**rules_victim, **rules_searcher}
            
        # action_prior = GreedyActionPrior(
        #     agent_id, set(rules.keys()), motion_policies[agent_id],
        #     rules, num_visits_init, big, look_after_move=look_after_move)
        # policy_model = PreferredPolicyModel(agent_role,
        #                                     action_prior,
        #                                     look_after_move=look_after_move)
        policy_model = PolicyModel(agent_id, agent_role, motion_policy=motion_policies[agent_id],
                                   look_after_move=look_after_move)

        # Construct reward model
        if agent_role == "searcher":
            reward_model = SearcherRewardModel(agent_id, role_to_ids=role_to_ids, big=big, small=small)
        elif agent_role == "victim":
            reward_model = VictimRewardModel(agent_id, role_to_ids=role_to_ids, big=big, small=small)
        elif agent_role == "suspect":
            sensors = kwargs.get("sensors", {})
            reward_model = SuspectRewardModel(agent_id, sensors, role_to_ids=role_to_ids, big=big, small=small)

        # Construct initial belief (histogram)
        init_belief = SARAgent.init_histogram_belief(agent_id, prior, role_to_ids, grid_map)
        return SARAgent(agent_id, init_belief, policy_model,
                        transition_model, observation_model, reward_model)
    

    @classmethod
    def init_histogram_belief(self, agent_id, prior, role_to_ids, grid_map):
        oo_hists = {}
        for role in role_to_ids:
            for id2 in role_to_ids[role]:
                hist = {}
                total_prob = 0
                if id2 in prior:
                    for pose in prior[id2]:
                        state = SARAgent.build_state(id2, role, pose)
                        hist[state] = prior[id2][pose]
                        total_prob += hist[state]                        
                else:
                    for free_loc in grid_map.free_poses:
                        if role == "target":
                            state = TargetState(id2, free_loc)
                        else:
                            for th in MotionAction.ORIENTATIONS:
                                pose = (*free_loc, th)
                                state = SARAgent.build_state(id2, role, pose)
                        hist[state] = 1.0
                        total_prob += hist[state]
                for state in hist:
                    hist[state] /= total_prob

                oo_hists[id2] = pomdp_py.Histogram(hist)
        return SARAgentBelief(agent_id, oo_hists)

    @staticmethod
    def build_state(id2, role, pose):
        if role == "searcher":
            state = SearcherState(id2, pose, (), False, time=-1)
        elif role == "victim":
            state = VictimState(id2, pose, (), False, time=-1)
        elif role == "suspect":
            state = SuspectState(id2, pose, (), False, time=-1)
        elif role == "target":
            state = TargetState(id2, pose[:2])
        return state


class SARAgentBelief(pomdp_py.OOBelief):
    """This is needed to make sure the belief is sampling the right
    type of State for this problem."""
    def __init__(self, agent_id, object_beliefs):
        """
        robot_id (int): The id of the robot that has this belief.
        object_beliefs (objid -> GenerativeDistribution)
            (includes robot)
        """
        self.agent_id = agent_id
        super().__init__(object_beliefs)

    def mpe(self, **kwargs):
        return JointState(pomdp_py.OOBelief.mpe(self, **kwargs).object_states)

    def random(self, **kwargs):
        return JointState(pomdp_py.OOBelief.random(self, **kwargs).object_states)
    

####### UNIT TESTS #########
worldstr=\
"""
Rx...
.x.xV
.S...    
"""
def unittest():
    from search_and_rescue.models.grid_map import GridMap
    from search_and_rescue.models.sensor import Laser2DSensor
    from search_and_rescue.env.env import unittest as env_unittest

    env, role_to_ids, sensors, motion_actions, look_after_move =\
        env_unittest(worldstr)
    for role in role_to_ids:
        for agent_id in env.ids_for(role):
            agent = SARAgent.construct(agent_id, role, sensors[agent_id],
                                       role_to_ids, env.grid_map, motion_actions, look_after_move=look_after_move,
                                       prior={agent_id: {env.state.pose(agent_id):1.0}})
            print("agent %d own state:" % agent_id)
            print(agent.belief.mpe().object_states[agent_id])
    

if __name__ == '__main__':
    unittest()
