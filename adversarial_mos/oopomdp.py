import pomdp_py
from sciex import Trial
import random
import copy
from dynamic_mos.env.env import *
from dynamic_mos.env.visual import *
from dynamic_mos.domain.action import *
from dynamic_mos.models.components.motion_policy import *
from dynamic_mos.experiments.world_types import *
from dynamic_mos.experiments.baselines.handcraft import *
from adversarial_mos import *
from dynamic_mos.dynamic_worlds import *
import concurrent.futures
import time
import cv2
import sys
import os
random.seed(1000)


class ParallelPlanner(pomdp_py.Planner):
    def __init__(self, planners, agents, robot_id):
        self._planners = planners
        self._agents = agents
        self._robot_id = robot_id

    @property
    def planners(self):
        return self._planners

    @property
    def agents(self):
        return self._agents

    def plan(self, agent_ids):
        # Plan with all planners
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self._plan_single, agent_ids)
        # results = []
        # for agent_id in self._planners:
        #     results.append(self._plan_single(agent_id))

        # Return a composite action
        actions = {}
        for res in results:
            if res is not None:
                agent_id, action, action_value, agent, num_sims = res
                actions[agent_id] = action
                print("Agent %d: %s" % (agent_id, str(action)))
        return CompositeAction(actions)

    def _plan_single(self, agent_id):
        # if agent_id != self._robot_id:
        #     return None
        action = self._planners[agent_id].plan(self._agents[agent_id])
        if isinstance(self._planners[agent_id], pomdp_py.POUCT):
            action_value = self._agents[agent_id].tree[action].value
            last_num_sims = self._planners[agent_id].last_num_sims
        else:
            action_value = 0
            last_num_sims = 0
        return agent_id, action, action_value, self._agents[agent_id], last_num_sims

    def _update_belief(self, tup):
        agent_id, action, observation, next_agent_state, agent_state = tup
        agent = self._agents[agent_id]
        for objid in agent.cur_belief.object_beliefs:
            if objid == agent_id:
                new_belief = pomdp_py.Histogram({next_agent_state: 1.0})
            else:
                if isinstance(next_agent_state, RobotState)\
                   and objid in next_agent_state["objects_found"]:
                    continue
                belief_obj = agent.cur_belief.object_belief(objid)
                if isinstance(agent.observation_model, pomdp_py.OOObservationModel):
                    obj_observation = observation.for_obj(objid)
                    observation_model = agent.observation_model[objid]
                else:
                    assert observation.objid == self._robot_id
                    obj_observation = observation
                    observation_model = agent.observation_model

                next_state_space = set({})
                for state in belief_obj:
                    next_state = copy.deepcopy(state)
                    if "time" in next_state.attributes:
                        next_state["time"] = state["time"] + 1
                    next_state_space.add(next_state)         

                new_belief = pomdp_py.update_histogram_belief(
                    belief_obj,
                    action, obj_observation,
                    observation_model,
                    agent.transition_model[objid],
                    next_state_space=next_state_space,
                    targs={"robot_state": agent_state},
                    oargs={"next_robot_state": next_agent_state})

            agent.cur_belief.set_object_belief(objid, new_belief)
        return (agent_id, agent.cur_belief)
            
    def update(self, real_action, real_observation, next_state, state, agent_ids=None):
        assert isinstance(real_action, CompositeAction)
        assert isinstance(real_observation, CompositeObservation)

        if agent_ids is None:
            agent_ids = set(self._agents.keys())        

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(
                self._update_belief,
                ((agent_id, real_action[agent_id], real_observation[agent_id],
                  next_state.object_states[agent_id], state.object_states[agent_id])
                 for agent_id in agent_ids))
        for agent_id, belief in results:
            # if agent_id != self._robot_id:
            #     continue
            self._agents[agent_id].set_belief(belief)


        # for agent_id in agent_ids:
        #     self._update_belief((agent_id, real_action[agent_id],
        #                          real_observation[agent_id],
        #                          next_state.object_states[agent_id],
        #                          state.object_states[agent_id]))
        
        for agent_id in agent_ids:
            self._agents[agent_id].update_history(real_action[agent_id], real_observation[agent_id])
            self._planners[agent_id].update(self._agents[agent_id], real_action[agent_id], real_observation[agent_id])            


class AdversarialTrial(Trial):

    def __init__(self, name, config, verbose=False):
        super().__init__(name, config, verbose=verbose)


    def _do_loop(self, env, planning_agent_ids, ma_planner, all_agent_ids):
        comp_action = ma_planner.plan(planning_agent_ids)
        prev_state = copy.deepcopy(env.state)
        reward = env.state_transition(comp_action, execute=True)
        comp_observation = CompositeObservation({
            agent_id:
            env.provide_observation(
                ma_planner.agents[agent_id].observation_model,
                comp_action[agent_id])
            for agent_id in ma_planner.agents
        })
        ma_planner.update(comp_action, comp_observation, copy.deepcopy(env.state),
                          prev_state, agent_ids=all_agent_ids)
        return comp_action, comp_observation, reward, prev_state

    def _do_viz(self, viz, env, viz_state, robot_id,
                agents, comp_action, comp_observation,
                look_after_move=True, save_path=None, step_index=-1):
        # Visualize
        robot_pose = env.state.object_states[robot_id].pose
        viz_observation = MosOOObservation({})
        if isinstance(comp_action[robot_id], LookAction)\
           or isinstance(comp_action[robot_id], FindAction)\
           or look_after_move:
            viz_observation = \
                agents[robot_id].sensor.observe(robot_pose,
                                                viz_state)
        viz.update(robot_id,
                   comp_action[robot_id],
                   comp_observation[robot_id],
                   viz_observation,
                   agents[robot_id].cur_belief)
        img = viz.on_render()
        if save_path is not None:
            # Rotate the image ccw 90 degree and convert color
            img = img.astype(np.float32)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate 90deg clockwise
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_path, "step_%d.png" % (step_index)), img)
        return img
        
    
    def run(self, logging=False):
        robot_char = "r"
        robot_id = interpret_robot_id(robot_char)        
        # mapstr, free_locations = create_connected_hallway_world(9, 1, 1, 3, 3) #create_free_world(6,6) #create_two_room_loop_world(5,5,3,1,1)#create_two_room_world(4,4,3,1)
        mapstr, free_locations = create_free_world(4,4) #create_two_room_loop_world(5,5,3,1,1)#create_two_room_world(4,4,3,1)        
        # mapstr, free_locations = create_two_room_loop_world(5,5,3,1,1)#create_two_room_world(4,4,3,1)        

        robot_pose = random.sample(free_locations, 1)[0]
        objD_pose = random.sample(free_locations - {robot_pose}, 1)[0]
        objE_pose = random.sample(free_locations - {robot_pose, objD_pose}, 1)[0]
        # robot_pose = (2, 1)
        # objE_pose = (3,0)

        # place objects
        mapstr = place_objects(mapstr,
                               {"r": robot_pose,
                                # "D": objD_pose,
                                "E": objE_pose})

        sensing_range = 3
        sensorstr = make_laser_sensor(90, (1, sensing_range), 0.5, False)
        worldstr = equip_sensors(mapstr, {robot_char: sensorstr})
        big = 100
        small = 1

        # interpret the world string
        dim, robots, objects, obstacles, sensors, _\
            = interpret(worldstr, {})

        # grid map
        grid_map = GridMap(dim[0], dim[1],
                           {objid: objects[objid].pose
                            for objid in obstacles})
        
        # init state
        init_state = MosOOState(robot_id, {robot_id: robots[robot_id],
                                           **objects})

        # motion policies
        motion_actions = create_motion_actions(scheme="xy", can_stay=True)
        target_objects = compute_target_objects(grid_map, init_state)
        motion_policies = {
            objid: BasicMotionPolicy(objid, grid_map, motion_actions)
            for objid in target_objects
        }

        # create env
        env = AdversarialMosEnvironment(init_state,
                                        robot_id,
                                        target_objects,
                                        sensors[robot_id],
                                        grid_map,
                                        motion_policies,
                                        big=big, small=small)

        # create agents
        agents = {}

        ## adversarial targets
        adv_prior = {}
        for x,y in free_locations:
            for th in MotionAction.ORIENTATIONS:
                if (x,y,th) == init_state.pose(robot_id):
                    adv_prior[(x,y,th)] = 1.0
                else:
                    adv_prior[(x,y,th)] = 1e-9
            
                
        for objid in env.target_objects:
            obj_sensor = copy.deepcopy(sensors[robot_id])
            obj_sensor.robot_id = objid
            target = AdversarialTarget(objid,
                                       init_state.object_states[objid],
                                       obj_sensor,
                                       motion_policies[objid],
                                       grid_map,
                                       robot_id,
                                       sensors[robot_id],
                                       prior={robot_id: adv_prior},
                                       action_prior=AdversarialActionPrior(objid, robot_id,
                                                                           grid_map, 10, big,
                                                                           motion_policies[objid]))
            agents[objid] = target

        agents[robot_id] = Searcher(robot_id,
                                    init_state.object_states[robot_id],
                                    sensors[robot_id],
                                    grid_map,
                                    env.target_objects,
                                    look_after_move=True,
                                    action_prior = DynamicMosActionPrior(robot_id, grid_map,
                                                                         10, big, look_after_move=True),
                                    motion_policies={objid: AdversarialPolicy(grid_map,
                                                                              sensors[robot_id].max_range,
                                                                              pr_stay=0.0,
                                                                              rule="avoid",
                                                                              motion_actions=motion_actions)
                                                     for objid in env.target_objects})

        # print(agents[2].observation_model.sample(env.state, None))
        # print(agents[5].observation_model.sample(env.state, None))
        # print(env.state.object_states[-114])
        # print(agents[-114].observation_model.sample(env.state, LookAction()))
        # print("BELIEF 2")
        # print(agents[2].belief.mpe())
        # print("BELIEF 5")
        # print(agents[5].belief.mpe())
        # print("BELIEF robot")
        # print(agents[-114].belief.mpe())
        # print("T robot")
        # print(agents[-114].transition_model.sample(env.state, MoveEast))
        # print(agents[-114].transition_model.sample(env.state, MoveWest))
        # print(agents[-114].transition_model.sample(env.state, MoveWest))
        # print("T 2")
        # import pdb; pdb.set_trace()
        # print(agents[2].transition_model.sample(env.state, MoveEast))
        # print(agents[5].transition_model.sample(env.state, MoveEast))
        
        
        # solve
        max_depth=10  # planning horizon
        discount_factor=0.95
        planning_time=1.       # amount of time (s) to plan each step
        exploration_const=1000
        max_steps=150
        look_after_move = True

        searcher_type = "pouct"
        
        planners = {
            aid: pomdp_py.POUCT(max_depth=max_depth,
                                  discount_factor=discount_factor,
                                  planning_time=planning_time,
                                  exploration_const=exploration_const,
                                  rollout_policy=agents[aid].policy_model,   # agent's policy model is preferred
                                  action_prior=agents[aid].policy_model.action_prior)
            for aid in agents
        }
        if searcher_type == "greedy":
            planners[robot_id] = GreedyPlanner(env.grid_map,
                                               look_after_move=look_after_move)
        if searcher_type == "random":
            planners[robot_id] = RandomPlanner(env.grid_map,
                                               look_after_move=look_after_move)

        ma_planner = ParallelPlanner(planners, agents, robot_id)

        # Visualization and saving of screenshots
        viz = MosViz(env, controllable=False)
        if viz.on_init() == False:
            raise Exception("Environment failed to initialize")
        viz_state = {}  # state that covers the whole map for observation visualization.
        for x in range(env.grid_map.width):
            for y in range(env.grid_map.length):
                viz_objid = len(viz_state)
                viz_state[viz_objid] = ObjectState(viz_objid, "obstacle", (x,y))
        viz_state = MosOOState(robot_id, viz_state)
        viz.update(robot_id,
                   None,
                   None,
                   None,
                   agents[robot_id].cur_belief)
        viz.on_render()
        save_path = None
        if len(sys.argv) > 1:
            save_path = sys.argv[1]
            if not os.path.exists(save_path):
                os.makedirs(save_path)        

        _find_actions_count = 0
        _total_reward = 0  # total, undiscounted reward        
        for i in range(max_steps):

            # Case 1: All plan in parallel
            comp_action, comp_observation, reward, prev_state\
                = self._do_loop(env, set(agents.keys()), ma_planner, set(agents.keys()))
            self._do_viz(viz, env, viz_state, robot_id, agents, comp_action, comp_observation,
                         look_after_move=look_after_move, save_path=save_path, step_index=i)

            # # Case 2: Robot plans first
            # print("ROBOT DO LOOP")
            # comp_action, comp_observation, reward, prev_state\
            #     = self._do_loop(env, {robot_id}, ma_planner, set(agents.keys()))
            # self._do_viz(viz, env, viz_state, robot_id, agents, comp_action, comp_observation,
            #              look_after_move=look_after_move)

            # _total_reward += reward
            # # Record find action count
            # if isinstance(comp_action[robot_id], FindAction):
            #     _find_actions_count += 1

            # print("TARGETS DO LOOP")
            # comp_action, comp_observation, _, prev_state\
            #     = self._do_loop(env, env.target_objects, ma_planner, set(agents.keys()))
            # self._do_viz(viz, env, viz_state, robot_id, agents, comp_action, comp_observation,
            #              look_after_move=look_after_move)                        
            
                
            # Info
            _step_info = "Step %d:  action: %s   reward: %.3f  cum_reward: %.3f"\
                % (i+1, str(comp_action[robot_id]), reward, _total_reward)
            if isinstance(ma_planner.planners[robot_id], pomdp_py.POUCT):
                _step_info += "   NumSims: %d" % ma_planner.planners[robot_id].last_num_sims
            if logging:
                trial_obj.log_event(Event("Trial %s | %s" % (trial_obj.name, _step_info)))
            else:
                print(_step_info)

            for target_id in env.target_objects:
                print("  Target %d believes robot is at %s" %
                      (target_id, str(agents[target_id].cur_belief.mpe().robot_state)))
            print("  [The robot is actually at %s]" % str(env.state.robot_state))

            # Termination check
            if set(env.state.object_states[robot_id].objects_found)\
               == env.target_objects:
                if logging:
                    trial_obj.log_event(Event("Trial %s | Task finished!\n\n" % (trial_obj.name)))
                break
            if _find_actions_count >= len(env.target_objects):
                if logging:
                    trial_obj.log_event(Event("Trial %s | Task ended; Used up Find actions.\n\n" % (trial_obj.name)))
                break
        
if __name__ == "__main__":
    trial = AdversarialTrial("a_1_b", {})
    trial.run()
