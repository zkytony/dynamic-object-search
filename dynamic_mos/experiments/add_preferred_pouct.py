# The job of this script is to add the baseline
# of pouct_preferred planner. This is meant to
# only be a fix temporarily. Next time you
# should remember to add that baseline.
from runner import *
import copy
import sciex
import sys
from pprint import pprint

def trial_func(global_name, seed, config):
    assert "problem_args" in config
    assert "solver_args" in config
    # Everything is the same except of the planner type and some args
    planner_type = "pouct#preferred"
    config_cpy = copy.deepcopy(config)
    config_cpy["solver_args"]["planner_type"] = planner_type
    config_cpy["problem_args"]["val_init"] = "big"
    config_cpy["problem_args"]["num_visits_init"] = 10
    world = config["world"]
    robot_char = world[1]
    sensor = config["problem_args"]["sensors"][robot_char]
    trial_name = "%s_%s" % (global_name, str(seed))
    kwargs = {**config_cpy["solver_args"], **config_cpy["problem_args"]}
    return make_trial(trial_name, world, sensor, **kwargs)

def main():
    if len(sys.argv) != 3:
        print("Usage: %s path-to-experiments split" % sys.argv[0])
        return
    
    sciex.add_baseline("pouct-preferred",
                       sys.argv[1],
                       trial_func,
                       save_trials=True,
                       split=int(sys.argv[2]))

if __name__ == "__main__":
    main()
