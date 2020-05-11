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
    num_trials = 30
    seeds = [random.randint(1, 1000000) for i in range(500)]
    scenarios = [((2, 2, 2, 1), 500, 60),
                 ((3, 3, 3, 1), 500, 120),
                 ((4, 4, 4, 1), 500, 180),
                 ((5, 5, 5, 1), 500, 240),
                 ((6, 6, 6, 1), 500, 300)]
    random.shuffle(scenarios)
    # Split the seeds into |scenarios| groups
    splitted_seeds = []
    for i in range(len(scenarios)):
        if (i+1)*num_trials > len(seeds):
            raise ValueError("Not enough seeds generated.")
        splitted_seeds.append(seeds[i*num_trials:(i+1)*num_trials])
    all_trials = []
    for i in range(len(scenarios)):
        world_case, max_steps, max_time = scenarios[i]
        for seed in splitted_seeds[i]:
            random.seed(seed)
            mapstr, free_locations = create_two_room_world(*world_case)
            robot_pose = random.sample(free_locations, 1)[0]
            objD_pose = random.sample(free_locations - {robot_pose}, 1)[0]
            objE_pose = random.sample(free_locations - {robot_pose, objD_pose}, 1)[0]
            world = (place_objects(mapstr,
                                   {"r": robot_pose,
                                    "D": objD_pose,
                                    "E": objE_pose}),
                     "r",
                     {"D": ("random", 0.4),
                      "E": ("random", 0.4)})
            trial_name = "domain%s_%s" % (str(scenarios[i]).replace(", ", "-"), str(seed))

            # Everything else use default
            big = 100
            small = 1
            params = {
                "big": big,
                "small": small,
                "exploration_const": 2*big,
                "max_time": max_time,
                "max_steps": max_steps,
                "visualize": VIZ,
                "planning_time": 0.7,
                "discount_factor": 0.99,
                "prior": "uniform"
            }

            
            # sensors
            sensors = [make_laser_sensor(90, (1, d), 0.5, False)
                       for d in {3, 4, 5, 6}]
            for sensor in sensors:
                pouct_trial = make_trial(trial_name, world, sensor, "pouct", **params)
                all_trials.append(pouct_trial)

    # Generate scripts to run experiments and gather results
    exp = Experiment("ScalabilityAA", all_trials, output_dir, verbose=True)
    exp.generate_trial_scripts(split=4)
    print("Find multiple computers to run these experiments.")

if __name__ == "__main__":
    main()
                
