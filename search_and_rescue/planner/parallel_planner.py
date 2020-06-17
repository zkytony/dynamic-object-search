import pomdp_py
import concurrent.futures
import copy
from search_and_rescue.env.action import ActionCollection
from search_and_rescue.env.observation import ObservationCollection

class ParallelPlanner(pomdp_py.Planner):
    def __init__(self, planners, agents):
        self._planners = planners
        self._agents = agents
        
    @property
    def planners(self):
        return self._planners

    @property
    def agents(self):
        return self._agents

    def plan(self, agent_ids):
        # # Plan with all planners
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self._plan_single, agent_ids)
        # results = []
        # for agent_id in self._planners:
        #     results.append(self._plan_single(agent_id))

        # Return a composite action
        actions = {}
        for res in results:
            if res is not None:
                agent_id, action, action_value, agent, num_sims = res
                actions[agent_id] = action
        return ActionCollection(actions)

    def _plan_single(self, agent_id):
        # if agent_id != self._robot_id:
        #     return None
        action = self._planners[agent_id].plan(self._agents[agent_id])
        if isinstance(self._planners[agent_id], pomdp_py.POUCT):
            action_value = self._agents[agent_id].tree[action].value
            last_num_sims = self._planners[agent_id].last_num_sims
        else:
            action_value = 0
            last_num_sims = 0
        return agent_id, action, action_value, self._agents[agent_id], last_num_sims

    def _update_belief(self, tup):
        agent_id, action, observation, next_agent_state, agent_state = tup
        agent = self._agents[agent_id]
        for objid in agent.cur_belief.object_beliefs:
            if objid == agent_id:
                new_belief = pomdp_py.Histogram({next_agent_state: 1.0})
            else:
                if hasattr(next_agent_state, "objects_found")\
                   and objid in next_agent_state["objects_found"]:
                    continue
                belief_obj = agent.cur_belief.object_belief(objid)
                if isinstance(agent.observation_model, pomdp_py.OOObservationModel):
                    obj_observation = observation.for_obj(objid)
                    observation_model = agent.observation_model[objid]

                next_state_space = set({})
                for state in belief_obj:
                    next_state = copy.deepcopy(state)
                    if "time" in next_state.attributes:
                        next_state["time"] = state["time"] + 1
                    next_state_space.add(next_state)

                new_belief = pomdp_py.update_histogram_belief(
                    belief_obj,
                    action, obj_observation,
                    observation_model,
                    agent.transition_model[objid],
                    next_state_space=next_state_space,
                    # The fact we're doing this makes using Histogram belief VERY WRONG.                    
                    targs={"agent_state": agent_state},
                    oargs={"next_agent_state": next_agent_state})

            agent.cur_belief.set_object_belief(objid, new_belief)
        return (agent_id, agent.cur_belief)
            
    def update(self, real_action, real_observation,
               next_state, state, agent_ids=None):
        assert isinstance(real_action, ActionCollection)
        assert isinstance(real_observation, ObservationCollection)

        if agent_ids is None:
            agent_ids = set(self._agents.keys())        

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(
                self._update_belief,
                ((agent_id, real_action[agent_id], real_observation[agent_id],
                  next_state.object_states[agent_id], state.object_states[agent_id])
                 for agent_id in agent_ids))
        for agent_id, belief in results:
            self._agents[agent_id].set_belief(belief)

        # for agent_id in agent_ids:
        #     self._update_belief((agent_id, real_action[agent_id],
        #                          real_observation[agent_id],
        #                          next_state.object_states[agent_id],
        #                          state.object_states[agent_id]))
        
        for agent_id in agent_ids:
            self._agents[agent_id].update_history(real_action[agent_id], real_observation[agent_id])
            self._planners[agent_id].update(self._agents[agent_id], real_action[agent_id], real_observation[agent_id])            

