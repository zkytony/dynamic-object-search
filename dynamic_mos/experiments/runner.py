from sciex import Experiment, Trial, Event, Result, YamlResult, PklResult, PostProcessingResult
from dynamic_mos import *
from dynamic_mos.dynamic_worlds import *
from dynamic_mos.experiments.result_types import *
from dynamic_mos.experiments.baselines.handcraft import *
from dynamic_mos.experiments.world_types import *
import copy
import time
import sys



class DynamicMosTrial(Trial):

    RESULT_TYPES = [RewardsResult, StatesResult, HistoryResult]

    def __init__(self, name, config, verbose=False):
        super().__init__(name, config, verbose=verbose)

    def run(self, logging=False):
        if logging:
            self.log_event(Event("Initializing problem instance of trial %s." % self.name))
        problem_args = self._config["problem_args"]
        solver_args = self._config["solver_args"]
        world = self._config["world"]
        _, robot_char, _ = world

        problem = DynamicMosOOPOMDP(robot_char,
                                    **problem_args)
        if logging:
            self.log_event(Event("Problem initialized.", kind=Event.SUCCESS))
        # Solve the problem
        return DynamicMosTrial.solve(problem,
                                     trial_obj=self,
                                     logging=logging,
                                     **solver_args)

    ### Solve the problem with POUCT/POMCP planner ###
    ### This is the main online POMDP solver logic ###
    @classmethod
    def solve(cls,
              problem,
              planner_type="pouct",
              max_depth=10,  # planning horizon
              discount_factor=0.99,
              planning_time=1.,       # amount of time (s) to plan each step
              exploration_const=1000, # exploration constant
              visualize=True,
              max_time=120,  # maximum amount of time allowed to solve the problem
              max_steps=500, # maximum number of planning steps the agent can take.
              save_path=None,  # path to directory to save screenshots for each step
              trial_obj=None,
              logging=False):
        """
        This function terminates when:
        - maximum time (max_time) reached; This time includes planning and updates
        - agent has planned `max_steps` number of steps
        - agent has taken n FindAction(s) where n = number of target objects.

        Args:
            visualize (bool) if True, show the pygame visualization.
        """
        robot_id = problem.agent.robot_id            
        if planner_type.startswith("pouct") and planner_type.endswith("preferred"):
            assert isinstance(problem.agent.policy_model, PreferredPolicyModel),\
                "Using pouct_preferred. Agent policy should be preferred policy model."
            planner = pomdp_py.POUCT(max_depth=max_depth,
                                     discount_factor=discount_factor,
                                     planning_time=planning_time,
                                     exploration_const=exploration_const,
                                     rollout_policy=problem.agent.policy_model,   # agent's policy model is preferred
                                     action_prior=problem.agent.policy_model.action_prior)
        elif planner_type.startswith("pouct"):
            # Use POUCT
            planner = pomdp_py.POUCT(max_depth=max_depth,
                                     discount_factor=discount_factor,
                                     planning_time=planning_time,
                                     exploration_const=exploration_const,
                                     rollout_policy=problem.agent.policy_model)  # Random by default
        elif planner_type == "pomcp":
            raise ValueError("Not supported for now.")
        elif planner_type == "pomcp_preferred":
            raise ValueError("Not supported for now.")
        elif planner_type.startswith("random"):
            planner = RandomPlanner(problem.env.grid_map,
                                    look_after_move=problem.look_after_move)
        elif planner_type.startswith("greedy"):
            planner = GreedyPlanner(problem.env.grid_map,
                                    look_after_move=problem.look_after_move)
        else:
            raise ValueError("Unsupported object belief type %s" % str(type(random_object_belief)))

        # Visualization initialize
        if visualize:
            viz = MosViz(problem.env, controllable=False)  # controllable=False means no keyboard control.
            if viz.on_init() == False:
                raise Exception("Environment failed to initialize")
            viz.update(robot_id,
                       None,
                       None,
                       None,
                       problem.agent.cur_belief)        
            viz.on_render()


        # Prepare recording results
        _Rewards = []
        _Values = []
        _States = [copy.deepcopy(problem.env.state)]
        _History = []

        # Start running
        _time_used = 0
        _find_actions_count = 0
        _total_reward = 0  # total, undiscounted reward
        for i in range(max_steps):
            # Plan action (timed)
            _start = time.time()
            real_action = planner.plan(problem.agent)
            _time_used += time.time() - _start
            if _time_used > max_time:
                break  # no more time to update.

            ## action value
            action_value = None
            if isinstance(planner, pomdp_py.POUCT):
                action_value = problem.agent.tree[real_action].value
                
            # Execute action
            robot_state = copy.deepcopy(problem.env.state.robot_state)
            reward = problem.env.state_transition(real_action, execute=True,
                                                  robot_id=robot_id)
            next_robot_state = copy.deepcopy(problem.env.state.robot_state)

            # Receive observation (timed)
            _start = time.time()
            real_observation = \
                problem.env.provide_observation(problem.agent.observation_model, real_action)

            # Updates (timed)
            problem.agent.clear_history()  # truncate history
            problem.agent.update_history(real_action, real_observation)
            belief_update(problem.agent, real_action, real_observation,
                          next_robot_state, robot_state, planner,
                          dynamic_object_ids=problem.env.dynamic_object_ids)
            _time_used += time.time() - _start

            # Add Reward
            _total_reward += reward

            # Record Find action count
            if isinstance(real_action, FindAction):
                _find_actions_count += 1            

            # Record other information
            _Rewards.append(reward)
            _States.append(copy.deepcopy(problem.env.state))
            _History += ((real_action, real_observation),)
            if isinstance(planner, pomdp_py.POUCT):
                _Values.append(action_value)            

            # Info
            _step_info = "Step %d:  action: %s   reward: %.3f  cum_reward: %.3f"\
                % (i+1, str(real_action), reward, _total_reward)            
            if isinstance(planner, pomdp_py.POUCT):
                _step_info += "   NumSims: %d" % planner.last_num_sims
            if logging:
                trial_obj.log_event(Event("Trial %s | %s" % (trial_obj.name, _step_info)))
            else:
                print(_step_info)

            # Visualize
            if visualize:
                # This is used to show the sensing range; Not sampled
                # according to observation model.
                robot_pose = problem.env.state.object_states[robot_id].pose
                viz_observation = MosOOObservation({})
                if isinstance(real_action, LookAction) or isinstance(real_action, FindAction):
                    viz_observation = \
                        problem.env.sensors[robot_id].observe(robot_pose,
                                                              problem.env.state)
                viz.update(robot_id,
                           real_action,
                           real_observation,
                           viz_observation,
                           problem.agent.cur_belief)
                viz.on_loop()
                img = viz.on_render()

                # Sleep a bit, if the planner is greedy or random
                if planner_type == "greedy" or planner_type == "random":
                    time.sleep(0.1)
                
                if save_path is not None:
                    # Rotate the image ccw 90 degree and convert color
                    img = img.astype(np.float32)
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate 90deg clockwise
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(save_path, "step_%d.png" % (i)), img)

            # Termination check
            if set(problem.env.state.object_states[robot_id].objects_found)\
               == problem.env.target_objects:
                if logging:
                    trial_obj.log_event(Event("Trial %s | Task finished!\n\n" % (trial_obj.name)))
                break
            if _find_actions_count >= len(problem.env.target_objects):
                if logging:
                    trial_obj.log_event(Event("Trial %s | Task ended; Used up Find actions.\n\n" % (trial_obj.name)))                
                break
            if _time_used > max_time:
                if logging:
                    trial_obj.log_event(Event("Trial %s | Task ended; Time limit reached.\n\n" % (trial_obj.name)))
                break

        results = [
            RewardsResult(_Rewards),
            StatesResult(_States),
            HistoryResult(_History),
        ]
        return results

def make_trial(trial_name, world, sensor, planner_type, **kwargs):
    grid_map_str, robot_char, motion_policies_dict = world
    problem_args = {"sigma": kwargs.get("sigma", 0.01),
                    "epsilon": kwargs.get("epsilon", 1.0),
                    "prior": kwargs.get("prior", "uniform"),
                    "agent_has_map": kwargs.get("agent_has_map", True),
                    "big": kwargs.get("big", 1000),
                    "small": kwargs.get("small", 1),
                    "sensors": {robot_char: sensor},
                    "motion_policies_dict": motion_policies_dict,
                    "grid_map_str": grid_map_str,
                    "look_after_move": kwargs.get("look_after_move", False)}
    if planner_type.endswith("preferred"):
        problem_args.update({"val_init": kwargs.get("val_init", "big"),
                             "num_visits_init": kwargs.get("num_visits_init", 10),
                             "use_preferred_policy": True})
    solver_args = {"planner_type": planner_type,
                   "max_depth": kwargs.get("max_depth", 10),
                   "discount_factor": kwargs.get("discount_factor", 0.99),
                   "planning_time": kwargs.get("planning_time", 0.7),
                   "exploration_const": kwargs.get("exploration_const", 100),
                   "max_time": kwargs.get("max_time", 120),
                   "max_steps": kwargs.get("max_steps", 500),
                   "visualize": kwargs.get("visualize", True)}
    sensor_str = sensor.replace(" ", ":").replace("_", "~")    
    return DynamicMosTrial("%s_%s-%s-%s" % (trial_name, planner_type,
                                            problem_args["prior"], sensor_str),
                           config={"problem_args": problem_args,
                                   "solver_args": solver_args,
                                   "world": world})


# Test
def unittest(world=None, planner_type="pouct", sensor_range=4, max_depth=20,
             planning_time=0.7, discount_factor=0.99, look_after_move=False,
             belief_rep="histogram"):
    # random world
    save_path = None
    if len(sys.argv) > 1:
        save_path = sys.argv[1]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    if world is None:
        world = dynamic_world_6
    grid_map_str, robot_char, motion_policies_dict = world
    laserstr = make_laser_sensor(90, (1, sensor_range), 0.5, False)
    proxstr = make_proximity_sensor(1, False)
    problem = DynamicMosOOPOMDP(robot_char,  # r is the robot character
                                sigma=0.01,  # observation model parameter
                                epsilon=1.0, # observation model parameter
                                grid_map_str=grid_map_str,
                                belief_rep=belief_rep,
                                sensors={robot_char: laserstr},
                                # TODO FIX
                                motion_policies_dict=motion_policies_dict,
                                prior="uniform",
                                agent_has_map=True,
                                big=100,
                                small=1,
                                use_preferred_policy=planner_type.endswith("preferred"),
                                val_init="big",
                                num_visits_init=10,
                                look_after_move=look_after_move)
    _total_reward = DynamicMosTrial.solve(problem,
                                          planner_type=planner_type,
                                          max_depth=max_depth,
                                          discount_factor=discount_factor,
                                          planning_time=planning_time,
                                          exploration_const=200,
                                          visualize=True,
                                          max_time=500,
                                          max_steps=150,
                                          save_path=save_path)
    return _total_reward

if __name__ == "__main__":
    unittest(dynamic_world_6, planner_type="pouct_preferred")
