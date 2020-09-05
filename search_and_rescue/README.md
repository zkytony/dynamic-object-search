# Project structure

### agent

```
agent.py: SARAgent (agent_id, b_0, Pi, T, O, R)
    Constructed through the SARAgent.construct() function.
```

### env

```
env.py: SAREnvironment (role_to_ids,
                        grid_map,
                        sensors,
                        s_0,
                        T,
                        R)
                        
env.py: MultiAgentRewardModel(R[i])
                        
                        
state.py: DynamicObjectState < ObjectState 
    must have "time" property
    
state.py: SearcherState < DynamicObjectState
    role is "searcher"
    
state.py: VictimState < DynamicObjectState
    role is "victim"
    
state.py: SuspectState < DynamicObjectState
    role is "suspect"
    
state.py: JointState < pomdp_py.OOState

state.py: EnvState < pomdp_py.State
    this is the state used by the Environment;
    It contains a list of "active agents" and a JointState.
    
    
observation.py: ObjectObservation(pomdp_py.Observation)

observation.py: JointObservation(pomdp_py.OOObservation)

observation.py: ObservationCollection(pomdp_py.Observation)
    Used to keep track of the observations made by multiple agents {map from agent id to observation}


action.py; MotionAction < Action (N/E/S/W)
action.py; LookAction < Action
action.py; FindAction < Action
action.py; ActionCollection < pomdp_py.Action
    Used to keep track of the actions taken by multiple agents (map from agent id to action)
    
```

### models

```
transition_model.py: StaticTransitionmodel
    for static objects
    
transition_model.py: DynamicAgenttransitionmodel
    for dynamic agents
    the dynamics is specified through a motion policy
    also needs a sensor
    a ~ Pi(s)

transition_model.py: DynamicObjecttransitionmodel
    the dynamic object doesn't plan its own actions
    Its dynamics is also specified through a motion policy
    
(There is no RobotTransitionModel because everything here are dynamic agents)

transition_model.py: JointTransitionModel(pomdp_py.OOTransitionModel)



sensor.py: Laser2DSensor < NoisySensor < GenerativeDistribution



observation_model.py: ObjectSensorModel < pomdp_py.ObservationModel
    This is basically a wrapper around the Sensor which already defines the distribution

observation_model.py: SensorModel < pomdp_py.OOObservationModel



policy_model.py: PolicyModel
    Random rollout policy
    
policy_model.py: PreferredPolicymodel
    Policy model with action prior 
    
policy_model.py: GreedyActionPrior
    Greedy action prior; Prefer motion acitons that moves agent according to
    greedy rules w.r.t. adversaries
    


motion_policy.py: StochasiticPolicy
    Generic stochasitic motion policy that pre-computes the valid motions given a grid map.
    Defines functionalities like find the path between two points.
    
motion_policy.py: BasicMotionPolicy
    Randomly take legal actions

motion_policy.py: AdversarialMotionPolicy
    Specify the agent's id and the adversary's id. If the rule is "avoid",
    avoid the adversary. If "chase", then chase. If "keep", then keep a distance
    or more. The agent has pr_stay probability of staying and otherwise takes
    an adversarial action.
    
motion_policy.py: MixedPolicy
    The agent is given several motion polciies and they are treated to weight equally.
```

# Thoughts

The structure of this project feels very good to me at the time I coded it up and everything worked,
and I was able to run a toy experiment against a human player.

However, I think the way the domain is set up is a little convoluted. It seems to
want to be super general (as many agents of any type as possible). But that I think
would actually backfire. 

The nice thing about this I remember, is that the belief of each agent is plotted separately
using matplotlib. It was really cool. 




