from sciex import Trial, Event
from search_and_rescue import *
from search_and_rescue.planner.parallel_planner import *

class SARTrial(Trial):
    def __init__(self, name, config, verbose=False):
        super().__init__(name, config, verbose=verbose)

    def run(self, logging=False):
        problem_args = self._config["problem_args"]
        solver_args = self._config["solver_args"]
        worldstr = self._config["world"]

        # Parameters
        can_stay = problem_args.get("can_stay", True)
        look_after_move = problem_args.get("look_after_move", True)
        
        # Building environment
        dims, robots, objects, obstacles, sensors, role_to_ids = interpret(worldstr)
        grid_map = GridMap(dims[0], dims[1],
                           {objid: objects[objid].pose
                            for objid in obstacles})
        motion_actions = create_motion_actions(can_stay=can_stay)
        env = SAREnvironment.construct(role_to_ids,
                                       {**robots, **objects},
                                       grid_map, motion_actions, sensors,
                                       look_after_move=look_after_move)        

        agents = {}
        for role in role_to_ids:
            for agent_id in env.ids_for(role):
                if "prior" in problem_args and agent_id in problem_args["prior"]:
                    prior = problem_args["prior"]
                else:
                    prior = {agent_id: {env.state.pose(agent_id):1}}
                agent = SARAgent.construct(agent_id, role, sensors[agent_id],
                                           role_to_ids, env.grid_map, motion_actions, look_after_move=look_after_move,
                                           prior=prior)
                agents[agent_id] = agent

        self._solve(env, agents, solver_args, logging=logging)
        
    def _solve(self, env, agents, solver_args, logging=False):
        # Parameters
        max_depth = solver_args.get("max_depth", 10)
        discount_factor = solver_args.get("discount_factor", 0.99)
        planning_time = solver_args.get("planning_time", 1.)
        exploration_const = solver_args.get("exploration_const", 200)
        visualize = solver_args.get("visualize", False)
        max_time = solver_args.get("max_time", 500)
        max_steps = solver_args.get("max_steps", 150)
        save_path = solver_args.get("save_path", None)
        controller_id = solver_args.get("controller_id", None)        

        # Build planners
        all_planners = {}
        for aid in agents:
            planner = pomdp_py.POUCT(
                discount_factor=discount_factor,
                planning_time=planning_time,
                exploration_const=exploration_const,
                rollout_policy=agents[aid].policy_model,   # agent's policy model is preferred
                action_prior=agents[aid].policy_model.action_prior)
            all_planners[aid] = planner
        ma_planner = ParallelPlanner(all_planners, agents)

        if visualize:
            viz, viz_state = self._init_viz(env, agents, controller_id)

        _find_actions_count = 0
        _total_reward = {aid: 0 for aid in agents}  # total, undiscounted reward        
        for i in range(max_steps):

            # Case 1: All plan in parallel
            comp_action, comp_observation, reward, prev_state\
                = self._do_loop(env, set(agents.keys()), ma_planner, set(agents.keys()))

            if visualize:
                self._do_viz(env, agents, viz, viz_state,
                             comp_action, comp_observation)
            
            # Record
            for aid in agents:
                _total_reward[aid] += reward[aid]

            # Info
            _step_info = self._do_info(i, comp_action, reward, _total_reward, ma_planner)
            if logging:
                self.log_event(Event("Trial %s | %s" % (self.name, _step_info)))
            else:
                print(_step_info)

            # Termination check
            active_roles = {env.role_for(aid) for aid in env.state.active_agents}
            if "victim" not in active_roles\
               and len(env.ids_for("victim")) > 0:
                # Victims have been caught; Suspects won
                if logging:
                    self.log_event(Event("Trial %s | Task ended; Suspects Won.\n\n" % (self.name)))
                break
            if "suspect" not in active_roles\
               and len(env.ids_for("suspect")) > 0:
                # Suspects have been caught. Searcher won
                if logging:
                    self.log_event(Event("Trial %s | Task ended; Searcher Won.\n\n" % (self.name)))
                break

    def _init_viz(self, env, agents, controller_id=None):
        # A state built for visualizing the fOV
        viz_state = {}
        z_viz = {}  # shows the whole FOV
        for x in range(env.grid_map.width):
            for y in range(env.grid_map.length):
                viz_objid = len(viz_state)
                viz_state[viz_objid] = ObstacleState(viz_objid, (x,y))
        viz_state = JointState(viz_state)

        viz = SARViz(env, controller_id=controller_id)
        if viz.on_init() == False:
            raise Exception("Visualization failed to initialize")        
        img = viz.on_render()        
        return viz, viz_state
    
    def _do_viz(self, env, agents, viz, viz_state,
                comp_action, comp_observation):
        # Sample observation
        for aid in agents:
            z_viz = {}      # shows only relevant object
            viz_state.set_object_state(aid, copy.deepcopy(env.state.object_states[aid]))
            for objid in viz_state.object_states:
                z_viz[objid] = env.sensors[aid].random(viz_state,
                                                       comp_action,
                                                       object_id=objid).pose
            observation_fov = JointObservation(z_viz)
            viz.update(aid, comp_action[aid], comp_observation[aid], observation_fov, None)
        img = viz.on_render()

    def _do_loop(self, env, planning_agent_ids, ma_planner, all_agent_ids):
        comp_action = ma_planner.plan(planning_agent_ids)
        prev_state = copy.deepcopy(env.state)
        reward = env.state_transition(comp_action, execute=True)
        comp_observation = ObservationCollection({
            agent_id:
            env.provide_observation(
                ma_planner.agents[agent_id].observation_model,
                comp_action[agent_id])
            for agent_id in ma_planner.agents
        })
        ma_planner.update(comp_action, comp_observation, copy.deepcopy(env.state),
                          prev_state, agent_ids=all_agent_ids)
        return comp_action, comp_observation, reward, prev_state

    def _do_info(self, i, comp_action, reward, _total_reward, ma_planner):
        _step_info = "Step %d:\n" % (i+1)
        for agent_id in ma_planner.planners:
            _step_info += "    Action: %s    Reward: %.3f    Cumulative Reward: %.3f"\
                % (str(comp_action[agent_id]), reward[agent_id], _total_reward[agent_id])
            if isinstance(ma_planner.planners[agent_id], pomdp_py.POUCT):
                _step_info += "   NumSims: %d" % ma_planner.planners[agent_id].last_num_sims
            _step_info += "\n"
        return _step_info



def unittest():
    from dynamic_mos.experiments.world_types import create_free_world
    from dynamic_mos.example_worlds import place_objects

    # Create world
    mapstr, free_locations = create_free_world(8,8)#create_connected_hallway_world(9, 1, 1, 3, 3) # #create_two_room_loop_world(5,5,3,1,1)#create_two_room_world(4,4,3,1)
    searcher_pose = random.sample(free_locations, 1)[0]
    victim_pose = random.sample(free_locations - {searcher_pose}, 1)[0]
    suspect_pose = random.sample(free_locations - {victim_pose, searcher_pose}, 1)[0]    
    laserstr = make_laser_sensor(90, (1, 2), 0.5, False)    
    mapstr = place_objects(mapstr,
                           {"R": searcher_pose,
                            "V": victim_pose,
                            "S": suspect_pose})
    worldstr = equip_sensors(mapstr, {"S": laserstr,
                                      "V": laserstr,
                                      "R": laserstr})
    problem_args = {}
    solver_args = {"visualize": True}
    config = {"problem_args": problem_args,
              "solver_args": solver_args,
              "world": worldstr}

    trial = SARTrial("trial_0_test", config, verbose=True)
    trial.run(logging=True)


if __name__ == '__main__':
    unittest()
