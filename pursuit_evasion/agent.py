from pursuit_evasion.domain import *
from pursuit_evasion.models import *
from pursuit_evasion.env import *
import pomdp_py
import pandas as pd

class Agent(pomdp_py.Agent):

    def __init__(self,
                 role,
                 init_belief,
                 opponent_rational=False):
        transition_model = TransitionModel()
        observation_model = ObservationModel(role)
        reward_model = RewardModel(role, opponent_rational=opponent_rational)
        policy_model = PolicyModel(role)
        self.role = role
        super().__init__(init_belief,
                         policy_model,
                         transition_model=transition_model,
                         observation_model=observation_model,
                         reward_model=reward_model)

def print_belief(role, belief):
    self_state = belief.mpe().for_role(role)
    print("    self at: ", self_state)
    for state in belief:
        if not(state.for_role(role) == self_state):
            continue
        if role == "pursuer":
            print("    evader at ", state.estate, belief[state])
        else:
            print("    pursuer at ", state.pstate, belief[state])

def merge(b1, b2):
    res = {}
    tot = 0
    for state in b1:
        res[state] = b1[state] + b2[state]
        tot += res[state]
    for state in res:
        res[state] /= tot
    return pomdp_py.Histogram(res)


def run_trial(pstate, estate, opponent_rational=False):
    state = JointState(pstate, estate)
    env = PursuitEvasionEnvironment(state)

    other_pstate = PState(other_side(pstate.side))
    other_estate = EState(other_side(estate.side))    
    
    bp0 = pomdp_py.Histogram({JointState(pstate, estate):0.5,
                              JointState(pstate, other_estate):0.5,
                              JointState(other_pstate, estate):1e-12,
                              JointState(other_pstate, other_estate):1e-12})
    be0 = pomdp_py.Histogram({JointState(pstate, estate):0.5,
                              JointState(other_pstate, estate):0.5,
                              JointState(pstate, other_estate):1e-12,
                              JointState(other_pstate, other_estate):1e-12})
    pursuer = Agent("pursuer", bp0, opponent_rational=opponent_rational)
    evader = Agent("evader", be0, opponent_rational=opponent_rational)
    print(pursuer.belief.random())

    discount_factor = 0.95
    pouct_pursuer = pomdp_py.POUCT(max_depth=10,
                                   discount_factor=discount_factor,
                                   planning_time=1.0,
                                   exploration_const=200,
                                   rollout_policy=evader.policy_model)
        
    pouct_evader = pomdp_py.POUCT(max_depth=10,
                                  discount_factor=discount_factor,
                                  planning_time=1.0,
                                  exploration_const=200,
                                  rollout_policy=evader.policy_model)
    print(env)
    print_belief("pursuer", pursuer.belief)
    print_belief("evader", evader.belief)
    print()

    cum_reward = {"pursuer": 0.0,
                  "evader": 0.0}
    discount = 1.0
    for step in range(20):
        print("\n---Step %d---" % step)
        ap = pouct_pursuer.plan(pursuer)
        paction = ap.paction
        paction_value = pursuer.tree[ap].value
        
        ae = pouct_evader.plan(evader)
        eaction = ae.eaction
        eaction_value = evader.tree[ae].value

        action = JointAction(paction, eaction)
        reward = env.state_transition(action, execute=True)
        op = pursuer.observation_model.sample(env.state, ap)
        oe = evader.observation_model.sample(env.state, ae)        

        bnew_p = pomdp_py.update_histogram_belief(pursuer.belief,
                                                  ap, op,
                                                  pursuer.observation_model,
                                                  pursuer.transition_model)
        bnew_e = pomdp_py.update_histogram_belief(evader.belief,
                                                  ae, oe,
                                                  evader.observation_model,
                                                  evader.transition_model)

        pursuer.set_belief(bnew_p)
        evader.set_belief(bnew_e)
        pouct_pursuer.update(pursuer, ap, op)
        pouct_evader.update(evader, ae, oe)

        print(env)        
        print("Real actions (P,E): (%s, %s); (%.3f, %.3f)" % (paction, eaction, paction_value, eaction_value))
        print("   P guess E, E guess P: (%s, %s)" % (ap.eaction, ae.paction))
        print("   observation (P): %s" % op)
        print("   observation (E): %s" % oe)
        print("Updated belief (P):")
        print_belief("pursuer", pursuer.belief)
        print("Updated belief (E):")
        print_belief("evader", evader.belief)            
        print(reward)
        print("\_______________________")

        cum_reward["pursuer"] += discount*reward["pursuer"]
        cum_reward["evader"] += discount*reward["evader"]
        discount *= discount_factor

        if paction.name == "open-door"\
            or eaction.name == "open-door":
            break
    return cum_reward


if __name__ == "__main__":
    pLstate = PState("left-door")
    pRstate = PState("right-door")    
    eLstate = EState("left-door")
    eRstate = EState("right-door")

    all_combos = [(pLstate, eLstate),
                  (pLstate, eRstate),
                  (pRstate, eLstate),
                  (pRstate, eRstate)]

    for opponent_rational in [True, False]:
        rewards = {"pursuer":[],
                   "evader":[]}
        for trial in range(30):
            pstate, estate = random.choice(all_combos)
            cum_reward = run_trial(pstate, estate, opponent_rational=opponent_rational)
            rewards["pursuer"].append(cum_reward["pursuer"])
            rewards["evader"].append(cum_reward["evader"])
        df = pd.DataFrame(rewards)
        df.to_pickle("results-opprat=%s.pkl" % opponent_rational)


        
