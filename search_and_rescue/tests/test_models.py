from search_and_rescue.env.env import unittest as env_unittest
from search_and_rescue.agent.agent import unittest as agent_unittest
from search_and_rescue.models.motion_policy import unittest as mp_unittest
from search_and_rescue.models.observation_model import unittest as ob_unittest
from search_and_rescue.models.transition_model import unittest as tr_unittest
from search_and_rescue.models.reward_model import unittest as re_unittest
from search_and_rescue.models.policy_model import unittest as po_unittest

worldstr=\
"""
Rx...
.x.xV
.S...    
"""
def main():
    print("\n### motion policy test ###")
    mp_unittest()
    print("\n### observation model test ###")
    ob_unittest()
    print("\n### transition model test ###")
    tr_unittest()
    print("\n### reward model test ###")
    re_unittest()
    print("\n### policy model test ###")
    po_unittest()
    print("\n### environment test ###")
    env_unittest(worldstr)
    print("\n### agent test ###")
    agent_unittest()

if __name__ == "__main__":
    main()
