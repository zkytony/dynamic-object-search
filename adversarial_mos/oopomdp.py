import pomdp_py
from sciex import Trial
import random
import copy
from dynamic_mos.env.env import *
from dynamic_mos.env.visual import *
from dynamic_mos.domain.action import *
from dynamic_mos.models.components.motion_policy import *
from dynamic_mos.experiments.world_types import *
from adversarial_mos import *
from dynamic_mos.dynamic_worlds import *
import concurrent.futures


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

    def plan(self):
        # Plan with all planners
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self._plan_single, self._planners.keys())
        # results = []
        # for agent_id in self._planners:
        #     results.append(self._plan_single(agent_id))

        # Return a composite action
        actions = {}
        for agent_id, action, action_value, agent, num_sims in results:
            actions[agent_id] = action
        return CompositeAction(actions)

    def _plan_single(self, agent_id):
        action = self._planners[agent_id].plan(self._agents[agent_id])
        action_value = self._agents[agent_id].tree[action].value
        return agent_id, action, action_value, self._agents[agent_id], self._planners[agent_id].last_num_sims

    def _update_belief(self, tup):
        agent_id, action, observation, next_agent_state, agent_state = tup
        agent = self._agents[agent_id]
        for objid in agent.cur_belief.object_beliefs:
            if objid == agent_id:
                new_belief = pomdp_py.Histogram({next_agent_state: 1.0})
            else:
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
            
    def update(self, real_action, real_observation, next_state, state):
        assert isinstance(real_action, CompositeAction)
        assert isinstance(real_observation, CompositeObservation)        

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(
                self._update_belief,
                ((agent_id, real_action[agent_id], real_observation[agent_id],
                  next_state.object_states[agent_id], state.object_states[agent_id])
                 for agent_id in self._agents))

        for agent_id, belief in results:
            self._agents[agent_id].set_belief(belief)
        # for agent_id in self._agents:
        #     self._update_belief((agent_id, real_action[agent_id],
        #                          real_observation[agent_id],
        #                          next_state.object_states[agent_id],
        #                          state.object_states[agent_id]))
        
        for agent_id in self._agents:
            self._agents[agent_id].update_history(real_action[agent_id], real_observation[agent_id])


class AdversarialTrial(Trial):

    def __init__(self, name, config, verbose=False):
        super().__init__(name, config, verbose=verbose)


    def run(self, logging=False):

        robot_char = "r"
        robot_id = interpret_robot_id(robot_char)        
        mapstr, free_locations = create_two_room_world(5,5,5,1)

        robot_pose = random.sample(free_locations, 1)[0]
        objD_pose = random.sample(free_locations - {robot_pose}, 1)[0]
        objE_pose = random.sample(free_locations - {robot_pose, objD_pose}, 1)[0]        

        # place objects
        mapstr = place_objects(mapstr,
                               {"r": robot_pose,
                                "D": objD_pose,
                                "E": objE_pose})
        
        sensorstr = make_laser_sensor(90, (1, 3), 0.5, False)
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
        motion_actions = create_motion_actions(scheme="xy")        
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
        for objid in env.target_objects:
            obj_sensor = copy.deepcopy(sensors[robot_id])
            obj_sensor.robot_id = objid
            target = AdversarialTarget(objid,
                                       init_state.object_states[objid],
                                       obj_sensor,
                                       motion_policies[objid],
                                       grid_map,
                                       robot_id,
                                       prior={robot_id: {init_state.pose(robot_id): 1.0}},
                                       action_prior=AdversarialActionPrior(objid, robot_id,
                                                                           grid_map, 10, big,
                                                                           motion_policies[objid]))
            agents[objid] = target

        agents[robot_id] = Searcher(robot_id,
                                    init_state.object_states[robot_id],
                                    sensors[robot_id],
                                    grid_map,
                                    env.target_objects,
                                    action_prior = DynamicMosActionPrior(robot_id, grid_map,
                                                                         10, big, look_after_move=True),
                                    motion_policies={objid: AdversarialPolicy(grid_map,
                                                                              sensors[robot_id].max_range,
                                                                              pr_stay=0.0,
                                                                              rule="avoid",
                                                                              motion_actions=motion_actions)
                                                     for objid in objects})
        
        # solve
        max_depth=10  # planning horizon
        discount_factor=0.99
        planning_time=1.       # amount of time (s) to plan each step
        exploration_const=1000
        max_steps=150
        
        planners = {
            aid: pomdp_py.POUCT(max_depth=max_depth,
                                  discount_factor=discount_factor,
                                  planning_time=planning_time,
                                  exploration_const=exploration_const,
                                  rollout_policy=agents[aid].policy_model,   # agent's policy model is preferred
                                  action_prior=agents[aid].policy_model.action_prior)
            for aid in agents
        }

        ma_planner = ParallelPlanner(planners, agents, robot_id)
        
        viz = MosViz(env, controllable=False)
        if viz.on_init() == False:
            raise Exception("Environment failed to initialize")
        viz.update(robot_id,
                   None,
                   None,
                   None,
                   agents[robot_id].cur_belief)
        viz.on_render()

        _find_actions_count = 0
        _total_reward = 0  # total, undiscounted reward        
        for i in range(max_steps):

            # all planners plan in parallel
            comp_action = ma_planner.plan()

            # execute action
            prev_state = copy.deepcopy(env.state)
            reward = env.state_transition(comp_action, execute=True)

            # receive observation
            comp_observation = CompositeObservation({
                agent_id:
                env.provide_observation(
                    ma_planner.agents[agent_id].observation_model,
                    comp_action[agent_id])
                for agent_id in ma_planner.agents
            })

            # Update
            ma_planner.update(comp_action, comp_observation, copy.deepcopy(env.state), prev_state)
            
            _total_reward += reward

            # Record find action count
            if isinstance(comp_action[robot_id], FindAction):
                _find_actions_count += 1
                
            # Info
            _step_info = "Step %d:  action: %s   reward: %.3f  cum_reward: %.3f"\
                % (i+1, str(comp_action[robot_id]), reward, _total_reward)
            if isinstance(ma_planner.planners[robot_id], pomdp_py.POUCT):
                _step_info += "   NumSims: %d" % ma_planner.planners[robot_id].last_num_sims
            if logging:
                trial_obj.log_event(Event("Trial %s | %s" % (trial_obj.name, _step_info)))
            else:
                print(_step_info)

            # Visualize
            robot_pose = env.state.object_states[robot_id].pose
            viz_observation = MosOOObservation({})
            if isinstance(comp_action[robot_id], LookAction)\
               or isinstance(comp_action[robot_id], FindAction):
                viz_observation = \
                    agents[robot_id].sensor.observe(robot_pose,
                                                    env.state)
            viz.update(robot_id,
                       comp_action[robot_id],
                       comp_observation[robot_id],
                       viz_observation,
                       agents[list(target_objects)[0]].cur_belief)
            img = viz.on_render()

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
