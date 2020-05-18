from sciex import *
from dynamic_mos.experiments.runner import *
from result_types import *
import pomdp_py
import os
import random
from dynamic_mos import *

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

VIZ = False
output_dir = os.path.join(ABS_PATH, "results")

def main():
    num_trials = 25
    seeds = [random.randint(1, 1000000) for i in range(500)]

    max_steps = 150
    max_time = 500
    world_case = (8, 8, 8, 1)
    sensor = make_laser_sensor(90, (1, 4), 0.5, False)
    
    scenarios = [(0.01, 0.01),
                 (0.1, 0.1),
                 (0.3, 0.3),
                 (0.5, 0.5),
                 (0.7, 0.7),
                 (0.9, 0.9)]
    random.shuffle(scenarios)
    # Split the seeds into |scenarios| groups
    splitted_seeds = []
    for i in range(len(scenarios)):
        if (i+1)*num_trials > len(seeds):
            raise ValueError("Not enough seeds generated.")
        splitted_seeds.append(seeds[i*num_trials:(i+1)*num_trials])
    all_trials = []
    for i in range(len(scenarios)):
        pr_rnd1, pr_rnd2 = scenarios[i]
        for seed in splitted_seeds[i]:
            random.seed(seed)
            mapstr, free_locations = create_two_room_world(*world_case)
            robot_pose = random.sample(free_locations, 1)[0]
            objD_pose = random.sample(free_locations - {robot_pose}, 1)[0]
            objE_pose = random.sample(free_locations - {robot_pose, objD_pose}, 1)[0]
            objD_goal = random.sample(free_locations - {robot_pose, objD_pose, objE_pose}, 1)[0]
            objE_goal = random.sample(free_locations - {robot_pose, objD_pose, objE_pose}, 1)[0]            
            world = (place_objects(mapstr,
                                   {"r": robot_pose,
                                    "D": objD_pose,
                                    "E": objE_pose}),
                     "r",
                     {"D": ("goal", (objD_goal, pr_rnd1)),
                      "E": ("goal", (objE_goal, pr_rnd2))})
            trial_name = "epsgoal%s_%s" % (str(scenarios[i]).replace(", ", "-"), str(seed))

            # Everything else use default
            big = 100
            small = 1
            params = {
                "big": big,
                "small": small,
                "exploration_const": big,
                "max_time": max_time,
                "max_steps": max_steps,
                "visualize": VIZ,
                "planning_time": 0.7,
                "discount_factor": 0.95,
                "prior": "uniform",
                "max_depth": 40,
            }
            
            # sensors
            params["look_after_move"] = False
            random_trial = make_trial(trial_name, world, sensor, "random", **params)
            greedy_trial = make_trial(trial_name, world, sensor, "greedy", **params)
            pouct_trial = make_trial(trial_name, world, sensor, "pouct", **params)
            pouct_preferred_trial = make_trial(trial_name, world, sensor, "pouct#preferred", **params)
            all_trials.append(pouct_trial)
            all_trials.append(pouct_preferred_trial)
            all_trials.append(greedy_trial)
            all_trials.append(random_trial)

            params["look_after_move"] = True
            random_trial = make_trial(trial_name, world, sensor, "random#nolook", **params)
            greedy_trial = make_trial(trial_name, world, sensor, "greedy#nolook", **params)
            pouct_trial = make_trial(trial_name, world, sensor, "pouct#nolook", **params)
            pouct_preferred_trial = make_trial(trial_name, world, sensor, "pouct#nolook#preferred", **params)
            all_trials.append(pouct_trial)
            all_trials.append(pouct_preferred_trial)
            all_trials.append(greedy_trial)
            all_trials.append(random_trial)            

    # Generate scripts to run experiments and gather results
    exp = Experiment("EpsilonGoalAA", all_trials, output_dir, verbose=True)
    exp.generate_trial_scripts(split=9)
    print("Find multiple computers to run these experiments.")

if __name__ == "__main__":
    main()
                
