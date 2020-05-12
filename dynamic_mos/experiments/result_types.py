from sciex import Experiment, Trial, Event, Result, YamlResult, PklResult, PostProcessingResult
import numpy as np
import math
from dynamic_mos.domain.state import *
import os
import json

#### Actual results for experiments ####
class RewardsResult(YamlResult):
    def __init__(self, rewards):
        """rewards: a list of reward floats"""
        super().__init__(rewards)
    @classmethod
    def FILENAME(cls):
        return "rewards.yaml"

    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        # compute cumulative rewards
        myresult = {}
        for specific_name in results:
            all_rewards = []
            for seed in results[specific_name]:
                cum_reward = sum(list(results[specific_name][seed]))
                all_rewards.append(cum_reward)
            myresult[specific_name] = {'mean': np.mean(all_rewards),
                                       'std': np.std(all_rewards),
                                       '_size': len(results[specific_name])}
        return myresult


    @classmethod
    def save_gathered_results(cls, gathered_results, path):

        def _tex_tab_val(entry, bold=False):
            pm = "$\pm$" if not bold else "$\\bm{\pm}$"
            return "%.2f %s %.2f" % (entry["mean"], pm, entry["std"])

        # Save plain text
        with open(os.path.join(path, "rewards.txt"), "w") as f:
            json.dump(gathered_results, f, indent=4, sort_keys=True)

        return True


class StatesResult(PklResult):
    def __init__(self, states):
        """list of state objects"""
        super().__init__(states)
    
    @classmethod
    def FILENAME(cls):
        return "states.pkl"

    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        # Returns the number of objects detected at the end.
        myresult = {}
        for specific_name in results:
            all_counts = []
            for seed in results[specific_name]:
                result = results[specific_name][seed]
                for objid in result[-1].object_states:
                    if isinstance(result[-1].object_states[objid], RobotState):
                        count = len(result[-1].object_states[objid].objects_found)
                        all_counts.append(count)
                        break
            myresult[specific_name] = {'mean': np.mean(all_counts),
                                       'std': np.std(all_counts),
                                       '_size': len(all_counts)}
        return myresult

    @classmethod
    def save_gathered_results(cls, gathered_results, path):

        # Save latex table for this
        def _tex_tab_val(entry, bold=False):
            pm = "$\pm$" if not bold else "$\\bm{\pm}$"
            return "%.2f %s %.2f" % (entry["mean"], pm, entry["std"])
        
        with open(os.path.join(path, "detections.txt"), "w") as f:
            json.dump(gathered_results, f, indent=4, sort_keys=True)            

        return True


class HistoryResult(PklResult):
    def __init__(self, history):
        """list of state objects"""
        super().__init__(history)

    @classmethod
    def FILENAME(cls):
        return "history.pkl"    
