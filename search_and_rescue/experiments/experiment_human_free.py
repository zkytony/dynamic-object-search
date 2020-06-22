from sciex import *
import random
import copy
from search_and_rescue.experiments.trial import *
from dynamic_mos.experiments.world_types import create_free_world
from search_and_rescue.utils import place_objects

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(ABS_PATH, "results")

def main():
    num_trials = 15
    seeds = [random.randint(1, 1000000) for i in range(500)]

    CAN_STAY = True
    LOOK_AFTER_MOVE = True  # no look action
    WORLD_TYPE = "free"

    all_ai_trials = []
    all_human_trials = []
    for worldsize in [5, 10]:
        
        for i in range(num_trials):
            
            seed = seeds.pop()
            random.seed(seed)
            mapstr, free_locations = create_free_world(worldsize, worldsize)
            suspect_pose = random.sample(free_locations, 1)[0]
            searcher_pose = random.sample(free_locations - {suspect_pose}, 1)[0]
            laserstr = make_laser_sensor(60, (1,3), 0.5, False)
            mapstr = place_objects(mapstr,
                                   [("R", searcher_pose),
                                    ("S", suspect_pose)])
            worldstr = equip_sensors(mapstr, {"R": laserstr,
                                              "S": laserstr})

            problem_args = {"can_stay": CAN_STAY,
                            "look_after_move": LOOK_AFTER_MOVE}

            solver_args = {"planning_time": 0.7,
                           "exploration_const": 500,
                           "discount_factor": 0.95,
                           "max_steps": 60,
                           "max_depth": 30,
                           "greedy_searcher": False,
                           "controller_id": None}

            config = {"problem_args": problem_args,
                      "solver_args": solver_args,
                      "world": worldstr}

            # Searcher varies
            config["problem_args"]["mdp_agent_ids"] = {}
            config["solver_args"]["visualize"] = False
            config["solver_args"]["controller_id"] = None
            ai_pomdp_trial = SARTrial("free-searcher-vary_%d_ai#pomdp-%d" % (seed, worldsize),
                                      copy.deepcopy(config))
            config["problem_args"]["mdp_agent_ids"] = {7000}
            config["solver_args"]["visualize"] = False
            ai_mdp_trial = SARTrial("free-searcher-vary_%d_ai#mdp-%d" % (seed, worldsize),
                                    copy.deepcopy(config))
            config["problem_args"]["mdp_agent_ids"] = {}
            config["solver_args"]["controller_id"] = 7000
            config["solver_args"]["visualize"] = True
            human_trial = SARTrial("free-searcher-vary_%d_human-%d" % (seed, worldsize),
                                   copy.deepcopy(config))
            all_ai_trials.append(ai_pomdp_trial)
            all_ai_trials.append(ai_mdp_trial)            
            all_human_trials.append(human_trial)

            # Suspect varies
            config["problem_args"]["mdp_agent_ids"] = {}
            config["solver_args"]["visualize"] = False
            config["solver_args"]["controller_id"] = None
            ai_pomdp_trial = SARTrial("free-suspect-vary_%d_ai#pomdp-%d" % (seed, worldsize),
                                      copy.deepcopy(config))
            config["problem_args"]["mdp_agent_ids"] = {5000}
            config["solver_args"]["visualize"] = False
            ai_mdp_trial = SARTrial("free-suspect-vary_%d_ai#mdp-%d" % (seed, worldsize),
                                    copy.deepcopy(config))
            config["problem_args"]["mdp_agent_ids"] = {}
            config["solver_args"]["controller_id"] = 5000
            config["solver_args"]["visualize"] = True
            human_trial = SARTrial("free-suspect-vary_%d_human-%d" % (seed, worldsize),
                                   copy.deepcopy(config))
            all_ai_trials.append(ai_pomdp_trial)
            all_ai_trials.append(ai_mdp_trial)            
            all_human_trials.append(human_trial)

    print("Generating AI trials")
    exp_ai = Experiment("HumanVsAI", all_ai_trials, output_dir, verbose=True)
    exp_ai.generate_trial_scripts(prefix="run_ai", split=3)
    print("Generating Human trials")
    exp_human = Experiment(exp_ai.name, all_human_trials, output_dir,
                           verbose=True, add_timestamp=False)
    exp_human.generate_trial_scripts(prefix="run_human", split=1)
    print("Find multiple computers to run these experiments.")


if __name__ == "__main__":
    main()    
