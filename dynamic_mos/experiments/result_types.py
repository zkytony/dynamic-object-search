from sciex import Experiment, Trial, Event, Result, YamlResult, PklResult, PostProcessingResult
import numpy as np
from scipy import stats
import math
from dynamic_mos.domain.state import *
import os
import json
import matplotlib.pyplot as plt
import seaborn

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
            sample_size = len(results[specific_name])
            t_95 = stats.t.ppf(1-0.05, sample_size)
            myresult[specific_name] = {'mean': np.mean(all_rewards),
                                       'std': np.std(all_rewards),
                                       'conf-95': t_95*(np.std(all_rewards) / math.sqrt(sample_size)),
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
            
        # Create a plot
        if os.path.basename(path).lower().startswith("scalability"):
            plot_scalability(gathered_results,
                                     suffix="all-d3",
                                     planners={"random", "greedy", "pouct", "pouct#preferred"},
                                     sensor_ranges={3})
            plot_scalability(gathered_results,
                                     suffix="rp-d3",
                                     planners={"random", "pouct"},
                                     sensor_ranges={3})
            plot_scalability(gathered_results,
                                     suffix="rpg-d3",
                                     planners={"random", "pouct", "greedy"},
                                     sensor_ranges={3})
            plot_scalability(gathered_results,
                                     suffix="f-d3456",
                                     planners={"pouct#preferred"},
                                     sensor_ranges={3,4,5,6})
            plot_scalability(gathered_results,
                                     suffix="p-d3456",
                                     planners={"pouct"},
                                     sensor_ranges={3,4,5,6})
            plot_scalability(gathered_results,
                                     suffix="g-d3456",
                                     planners={"greedy"},
                                     sensor_ranges={3,4,5,6})
            plot_scalability(gathered_results,
                                     suffix="rpgf-d6",
                                     planners={"random", "greedy", "pouct", "pouct#preferred"},
                                     sensor_ranges={6})

        # Create a plot
        elif os.path.basename(path).lower().startswith("dynamics"):
            plot_dynamics(gathered_results,
                                  suffix="all-d4",
                                  planners={"random", "greedy", "pouct", "pouct#preferred"},
                                  sensor_ranges={4})

        return True


def plot_scalability(gathered_results,
                     suffix="plot", plot_type="rewards",
                     planners=None, sensor_ranges=None):
    # We plot reward vs the domain scale. Domain scale is represented
    # by a single integer n to indicate (n,n,n,1)
    colors = {"random": "red",
              "greedy": "green",
              "pouct": "blue",
              "pouct#preferred": "purple"}
    fmts = {"d3": "-",
            "d4": "--",
            "d5": "-.",
            "d6": ":"}
    fig = plt.figure()
    ax = plt.gca()
    xvals = []
    means = {}
    errs = {}
    for global_name in gathered_results:
        scenario = global_name.split("domain(")[1].split("_")[0][:-1]
        world_case = scenario.split(")-")[0][1:]
        domain_scale = int(world_case.split("-")[0])
        xvals.append(domain_scale)

        results = gathered_results[global_name]
        for specific_name in results:
            planner_type = specific_name.split("-")[0]
            sensor_str = specific_name.split("-")[2]
            max_range = int(sensor_str.split(":")[3].split("=")[-1])
            method_name = "%s-d%s" % (planner_type, max_range)
            if planners is not None and planner_type not in planners:
                continue
            if sensor_ranges is not None and max_range not in sensor_ranges:
                continue
            if method_name not in means:
                means[method_name] = {}
                errs[method_name] = {}
            means[method_name][domain_scale] = results[specific_name]["mean"]
            errs[method_name][domain_scale] = results[specific_name]["conf-95"]
    xvals = list(sorted(xvals))

    for method_name in sorted(means):
        planner_type = method_name.split("-")[0]
        range_str = method_name.split("-")[1]
        y_meth = [means[method_name][s] for s in xvals]
        yerr_meth = [errs[method_name][s] for s in xvals]
        plt.errorbar(xvals, y_meth, yerr=yerr_meth, label=method_name,
                     color=colors[planner_type],
                     fmt=fmts[range_str])
        
    plt.legend(loc="lower left")
    ax.set_title("Performance as Problem Scales")
    ax.set_xlabel("Domain Scale")
    ax.xaxis.set_ticks(xvals)
    if plot_type == "rewards":
        ax.set_ylabel("Cumulative Reward")
        fig.savefig("rewards-%s.png" % suffix)
    elif plot_type == "detections":
        ax.set_ylabel("Number of Detected Objects")
        fig.savefig("detections-%s.png" % suffix)



def plot_dynamics(gathered_results, plot_type="rewards",
                  suffix="plot", planners=None, sensor_ranges=None):
    # We plot reward vs the domain scale. Domain scale is represented
    # by a single integer n to indicate (n,n,n,1)
    colors = {"random": "red",
              "greedy": "green",
              "pouct": "blue",
              "pouct#preferred": "purple"}
    fmts = {"d3": "-",
            "d4": "--",
            "d5": "-.",
            "d6": ":"}
    fig = plt.figure()
    ax = plt.gca()
    xvals = []
    means = {}
    errs = {}
    for global_name in gathered_results:
        scenario = global_name.split("domain(")[1].split("_")[0][:-1]
        world_case = scenario.split(")-")[0][1:]
        dynamics = round(1.0-float(world_case.split("-")[1]), 2)
        xvals.append(dynamics)

        results = gathered_results[global_name]
        for specific_name in results:
            planner_type = specific_name.split("-")[0]
            sensor_str = specific_name.split("-")[2]
            max_range = int(sensor_str.split(":")[3].split("=")[-1])
            method_name = "%s-d%s" % (planner_type, max_range)
            if planners is not None and planner_type not in planners:
                continue
            if sensor_ranges is not None and max_range not in sensor_ranges:
                continue
            if method_name not in means:
                means[method_name] = {}
                errs[method_name] = {}
            means[method_name][dynamics] = results[specific_name]["mean"]
            errs[method_name][dynamics] = results[specific_name]["conf-95"]
    xvals = list(sorted(xvals))

    for method_name in sorted(means):
        planner_type = method_name.split("-")[0]
        range_str = method_name.split("-")[1]
        y_meth = [means[method_name][s] for s in xvals]
        yerr_meth = [errs[method_name][s] for s in xvals]
        plt.errorbar(xvals, y_meth, yerr=yerr_meth, label=method_name,
                     color=colors[planner_type],
                     fmt=fmts[range_str])
        
    plt.legend(bbox_to_anchor=(0., 0.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)        
    ax.set_title("Performance as Dynamic Increases (World Scale: 8)")
    ax.set_xlabel("Domain Scale")
    ax.xaxis.set_ticks(xvals)
    if plot_type == "rewards":
        ax.set_ylabel("Cumulative Reward")
        ax.set_ylim(bottom=-300)
        fig.savefig("rewards-%s.png" % suffix)
    elif plot_type == "detections":
        ax.set_ylabel("Number of Detected Objects")
        ax.set_ylim(bottom=-0.2)
        fig.savefig("detections-%s.png" % suffix)
    

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

        # Create a plot
        if os.path.basename(path).lower().startswith("scalability"):
            plot_scalability(gathered_results,
                             plot_type="detections",
                             suffix="all-d3",
                             planners={"random", "greedy", "pouct", "pouct#preferred"},
                             sensor_ranges={3})
            plot_scalability(gathered_results,
                             plot_type="detections",
                             suffix="rp-d3",
                             planners={"random", "pouct"},
                             sensor_ranges={3})
            plot_scalability(gathered_results,
                             plot_type="detections",
                             suffix="rpg-d3",
                             planners={"random", "pouct", "greedy"},
                             sensor_ranges={3})
            plot_scalability(gathered_results,
                             plot_type="detections",
                             suffix="f-d3456",
                             planners={"pouct#preferred"},
                             sensor_ranges={3,4,5,6})
            plot_scalability(gathered_results,
                             plot_type="detections",
                             suffix="p-d3456",
                             planners={"pouct"},
                             sensor_ranges={3,4,5,6})
            plot_scalability(gathered_results,
                             plot_type="detections",
                             suffix="g-d3456",
                             planners={"greedy"},
                             sensor_ranges={3,4,5,6})
            plot_scalability(gathered_results,
                             plot_type="detections",
                             suffix="rpgf-d6",
                             planners={"random", "greedy", "pouct", "pouct#preferred"},
                             sensor_ranges={6})

        # Create a plot
        elif os.path.basename(path).lower().startswith("dynamics"):
            plot_dynamics(gathered_results,
                          plot_type="detections",
                          suffix="all-d4",
                          planners={"random", "greedy", "pouct", "pouct#preferred"},
                          sensor_ranges={4})
            

        return True


class HistoryResult(PklResult):
    def __init__(self, history):
        """list of state objects"""
        super().__init__(history)

    @classmethod
    def FILENAME(cls):
        return "history.pkl"    
