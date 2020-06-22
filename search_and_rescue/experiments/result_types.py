from sciex import *
import numpy as np
from scipy import stats
import math
import os
import json
import yaml
import matplotlib.pyplot as plt
from search_and_rescue import *


#### Actual results for experiments ####
class RewardsResult(YamlResult):
    def __init__(self, rewards):
        """rewards: a list of reward floats"""
        self._rewards = rewards
        
    @classmethod
    def FILENAME(cls):
        return "rewards.yaml"

    def save(self, path):
        r = {}
        for comp_reward in self._rewards:
            for aid in comp_reward:
                if aid not in r:
                    r[aid] = []
                r[aid].append(comp_reward[aid])
        with open(path, "w") as f:
            yaml.dump(r, f)    

    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        # compute cumulative rewards
        myresult = {}
        for specific_name in results:
            all_rewards = {}
            for seed in results[specific_name]:
                rewards = results[specific_name][seed]
                for aid in rewards:
                    if aid not in all_rewards:
                        all_rewards[aid] = []
                    cum_reward = sum(list(rewards[aid]))
                    all_rewards[aid].append(cum_reward)
            sample_size = len(results[specific_name])
            t_95 = stats.t.ppf(1-0.05, sample_size)
            myresult[specific_name] = {}
            for aid in all_rewards:
                myresult[specific_name][aid] = {'mean': np.mean(all_rewards[aid]),
                                                'std': np.std(all_rewards[aid]),
                                                'conf-95': t_95*(np.std(all_rewards[aid]) / math.sqrt(sample_size)),
                                                '_size': sample_size}
        return myresult

    @classmethod
    def save_gathered_results(cls, gathered_results, path):

        def _tex_tab_val(entry, bold=False):
            pm = "$\pm$" if not bold else "$\\bm{\pm}$"
            return "%.2f %s %.2f" % (entry["mean"], pm, entry["std"])

        # Save plain text
        with open(os.path.join(path, "rewards.json"), "w") as f:
            json.dump(gathered_results, f, indent=4, sort_keys=True)

        plot_performance(gathered_results, plot_type="rewards")
            
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
                    if isinstance(result[-1].object_states[objid], SearcherState):
                        count = len(result[-1].object_states[objid].objects_found)
                        all_counts.append(count)
                        break
            sample_size = len(all_counts)
            t_95 = stats.t.ppf(1-0.05, sample_size)                    
            myresult[specific_name] = {'mean': np.mean(all_counts),
                                       'std': np.std(all_counts),
                                       'conf-95': t_95*(np.std(all_counts)/ math.sqrt(sample_size)),
                                       '_size': len(all_counts)}
        return myresult

    @classmethod
    def save_gathered_results(cls, gathered_results, path):

        # Save latex table for this
        def _tex_tab_val(entry, bold=False):
            pm = "$\pm$" if not bold else "$\\bm{\pm}$"
            return "%.2f %s %.2f" % (entry["mean"], pm, entry["std"])
        
        with open(os.path.join(path, "detections.json"), "w") as f:
            json.dump(gathered_results, f, indent=4, sort_keys=True)
            
        plot_performance(gathered_results, plot_type="detections")
        
        return True


class HistoryResult(PklResult):
    def __init__(self, history):
        """list of state objects"""
        super().__init__(history)

    @classmethod
    def FILENAME(cls):
        return "history.pkl"    


def plot_performance(gathered_results, suffix="plot", plot_type="rewards"):
    COLORS = {
        "ai-pomdp": "brown",
        "ai-mdp": "orange",
        "human": "green"
    }    
    for global_name in gathered_results:
        if plot_type == "rewards":
            title = "Searcher's Cumulative Reward \n vs."
        elif plot_type == "detections":
            title = "Number of Detections \n vs."
        if "searcher" in global_name:
            title += "Types of Searchers"
        elif "suspect" in global_name:
            title += "Types of Suspect (i.e. Adversarial Target)"
        results = gathered_results[global_name]
        
        fig = plt.figure()
        ax = plt.gca()
        xvals = []
        means = {}
        errs = {}

        for specific_name in results:
            if "#mdp" in specific_name:
                method_name = "ai-mdp"
            elif "#pomdp" in specific_name:
                method_name = "ai-pomdp"
            else:
                print("Unhandled method: %s" % specific_name)
                continue

            worldsize = int(specific_name.split("-")[1])
            if worldsize not in xvals:
                xvals.append(worldsize)
            if method_name not in means:
                means[method_name] = {}
                errs[method_name] = {}

            if plot_type == "rewards":
                means[method_name][worldsize] = results[specific_name][7000]["mean"]
                errs[method_name][worldsize] = results[specific_name][7000]["conf-95"]
            elif plot_type == "detections":
                means[method_name][worldsize] = results[specific_name]["mean"]
                errs[method_name][worldsize] = results[specific_name]["conf-95"]
            
        xvals = np.array(sorted(xvals))
        width = 0.4
        i = 0
        for method_name in sorted(means):
            # if method_name.startswith("keyword"):
            #     print("KEYWORD IGNORED.")
            #     continue
            y_meth = [means[method_name][s] for s in xvals]
            yerr_meth = [errs[method_name][s] for s in xvals]
            plt.bar(xvals + width*i, y_meth, width, yerr=yerr_meth, label=method_name,
                    color=COLORS[method_name])
            i += 1

        plt.legend(loc="lower left")
        ax.set_title(title)
        ax.set_xlabel("World Size")
        ax.set_xticks(xvals + width*(len(means)-1)/2)
        ax.set_xticklabels(xvals)
        if plot_type == "rewards":
            ax.set_ylabel("Cumulative Reward")
            fig.savefig("rewards-%s-%s.png" % (global_name, suffix))
        elif plot_type == "detections":
            ax.set_ylabel("Number of Detected Objects")
            fig.savefig("detections-%s-%s.png" % (global_name, suffix))
        plt.clf()
