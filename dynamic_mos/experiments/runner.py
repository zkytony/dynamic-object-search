from sciex import Experiment, Trial, Event, Result, YamlResult, PklResult, PostProcessingResult
from dynamic_mos import *


class DynamicMosTrial(Trial):

    def __init__(self, name, config, verbose=False):
        super().__init__(name, config, verbose=verbose)

    def run(self, logging=False):
        pass


def make_trial(trial_name,
               worldstr,    # tuple that defines the world
               sensorstr,
               sigma=0.01,  # observation model parameter
               epsilon=1.0): # observation model parameter               
    pass
