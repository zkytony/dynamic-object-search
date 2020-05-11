"""2D Multi-Object Search (MOS) Task.
Uses the domain, models, and agent/environment
to actually define the POMDP problem for multi-object search.
Then, solve it using POUCT or POMCP."""
import pomdp_py
from dynamic_mos.env.env import *
from dynamic_mos.env.visual import *
from dynamic_mos.agent.agent import *
from dynamic_mos.dynamic_worlds import *
from dynamic_mos.domain.observation import *
from dynamic_mos.models.components.grid_map import GridMap
from dynamic_mos.models.dynamic_transition_model import *
from dynamic_mos.experiments.runner import DynamicMosTrial
import argparse
import time
import random
import sys
import os
import copy

class DynamicMosOOPOMDP(pomdp_py.OOPOMDP):
    """
    A DynamicMosOOPOMDP is instantiated given a string
    description of the search world, sensor descriptions for robots,
    and the necessary parameters for the agent's models,
    a dictionary that maps from dynamic objects to motion policies.

    Note: This is of course a simulation, where you can
    generate a world and know where the target objects are
    and then construct the Environment object. But in the
    real robot scenario, you don't know where the objects
    are. In that case, as I have done it in the past, you
    could construct an Environment object and give None to
    the object poses.
    """
    def __init__(self, robot_char, env=None, grid_map_str=None,
                 sensors=None, sigma=0.01, epsilon=1,
                 belief_rep="histogram", prior={}, num_particles=100,
                 big=100, small=1,
                 agent_has_map=False,
                 motion_policies_dict={}):
        """
        Args:
            robot_char (int or str): the id of the agent that will solve this MosOOPOMDP.
                If it is a `str`, it will be interpreted as an integer using `interpret_robot_id`
                in env/env.py.
            env (MosEnvironment): the environment. 
            grid_map_str (str): Search space description. See env/env.py:interpret. An example:
                rx...
                .x.xT
                .....
                Ignored if env is not None
            sensors (dict): map from robot character to sensor string.
                For example: {'r': 'laser fov=90 min_range=1 max_range=5
                                    angle_increment=5'}
                Ignored if env is not None
            agent_has_map (bool): If True, we assume the agent is given the occupancy
                                  grid map of the world. Then, the agent can use this
                                  map to avoid planning invalid actions (bumping into things).
                                  But this map does not help the agent's prior belief directly.
            motion_policies_dict (dict): Maps from dynamic object string representation
                                         to a list of points the object traverses 
                                         (TODO: currently limited to iterative motion policy.)

            sigma, epsilon: observation model paramters
            belief_rep (str): belief representation. Either histogram or particles.
            prior (dict or str): either a dictionary as defined in agent/belief.py
                or a string, either "uniform" or "informed". For "uniform", a uniform
                prior will be given. For "informed", a perfect prior will be given.
            num_particles (int): setting for the particle belief representation
        """
        grid_map = None
        if env is None:
            assert grid_map_str is not None and sensors is not None,\
                "Since env is not provided, you must provide string descriptions"\
                "of the world and sensors."
            worldstr = equip_sensors(grid_map_str, sensors)
            # Interpret the world string
            dim, robots, objects, obstacles, sensors, motion_policies\
                = interpret(worldstr, motion_policies_dict)
            # Grid map
            grid_map = GridMap(dim[0], dim[1],
                               {objid: objects[objid].pose
                                for objid in obstacles})
            # Create motion policies for dynamic objects
            for objid in motion_policies:
                policy_type = motion_policies[objid][0]
                if policy_type == "iterative":
                    motion_policies[objid] =\
                        IterativeMotionPolicy(motion_policies[objid][1])
                elif policy_type == "random":
                    motion_policies[objid] =\
                        RandomStayPolicy(grid_map, motion_policies[objid][1])
                else:
                    raise ValueError("Unrecognized motion policy type")
            # Make init state
            init_state = MosOOState({**objects, **robots})
            env = MosEnvironment(dim,
                                 init_state, sensors,
                                 grid_map=grid_map,
                                 motion_policies=motion_policies)

        # construct prior
        if type(prior) == str:
            if prior == "uniform":
                prior = {}
            elif prior == "informed":
                prior = {}
                for objid in env.target_objects:
                    groundtruth_pose = env.state.pose(objid)
                    prior[objid] = {groundtruth_pose: 1.0}

        # Potential extension: a multi-agent POMDP. For now, the environment
        # can keep track of the states of multiple agents, but a POMDP is still
        # only defined over a single agent. Perhaps, MultiAgent is just a kind
        # of Agent, which will make the implementation of multi-agent POMDP cleaner.
        robot_id = robot_id if type(robot_char) == int else interpret_robot_id(robot_char)
        agent_grid_map = env.grid_map if agent_has_map else None
        agent = MosAgent(robot_id,
                         env.state.object_states[robot_id],
                         env.target_objects,
                         (env.width, env.length),
                         env.sensors[robot_id],
                         sigma=sigma,
                         epsilon=epsilon,
                         belief_rep=belief_rep,
                         prior=prior,
                         num_particles=num_particles,
                         grid_map=agent_grid_map,
                         motion_policies=env.dynamic_object_motion_policies,
                         small=small,
                         big=big)
        super().__init__(agent, env,
                         name="MOS(%d,%d,%d)" % (env.width, env.length, len(env.target_objects)))


# Test
def unittest(world=None):
    # random world
    save_path = None
    if len(sys.argv) > 1:
        save_path = sys.argv[1]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    if world is None:
        world = dynamic_world_6
    grid_map_str, robot_char, motion_policies_dict = world
    laserstr = make_laser_sensor(90, (1, 4), 0.5, False)
    proxstr = make_proximity_sensor(1, False)    
    problem = DynamicMosOOPOMDP(robot_char,  # r is the robot character
                                sigma=0.01,  # observation model parameter
                                epsilon=1.0, # observation model parameter
                                grid_map_str=grid_map_str,
                                sensors={robot_char: laserstr},
                                # TODO FIX
                                motion_policies_dict=motion_policies_dict,
                                prior="uniform",
                                agent_has_map=True,
                                big=100,
                                small=1)
    _total_reward = DynamicMosTrial.solve(problem,
                                          max_depth=20,
                                          discount_factor=0.95,
                                          planning_time=0.9,
                                          exploration_const=100,
                                          visualize=True,
                                          max_time=120,
                                          max_steps=500,
                                          save_path=save_path)
    return _total_reward

if __name__ == "__main__":
    unittest()

