from pursuit_evasion.domain import *
from pursuit_evasion.models import *
from pursuit_evasion.env import *
import pomdp_py

class Agent(pomdp_py.Agent):

    def __init__(self,
                 role,
                 init_belief):
        transition_model = TransitionModel()
        observation_model = ObservationModel(role)
        reward_model = RewardModel(role)
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

if __name__ == "__main__":
    pLstate = PState("left-door")
    pRstate = PState("right-door")    
    eLstate = EState("left-door")
    eRstate = EState("right-door")

    state = JointState(pLstate, eRstate)
    env = PursuitEvasionEnvironment(state)    
    
    bp0 = pomdp_py.Histogram({JointState(pLstate, eLstate):1e-12,
                              JointState(pLstate, eRstate):1.0,
                              JointState(pRstate, eLstate):1e-12,
                              JointState(pRstate, eRstate):1e-12})
    be0 = pomdp_py.Histogram({JointState(pLstate, eRstate):0.5,
                              JointState(pRstate, eRstate):0.5,
                              JointState(pLstate, eLstate):1e-12,
                              JointState(pRstate, eLstate):1e-12})
    pursuer = Agent("pursuer", bp0)
    evader = Agent("evader", be0)
    print(pursuer.belief.random())

    pouct_pursuer = pomdp_py.POUCT(max_depth=10,
                                   discount_factor=0.95,
                                   planning_time=1.0,
                                   exploration_const=200,
                                   rollout_policy=evader.policy_model)
        
    pouct_evader = pomdp_py.POUCT(max_depth=10,
                                  discount_factor=0.95,
                                  planning_time=1.0,
                                  exploration_const=200,
                                  rollout_policy=evader.policy_model)
    print(env)
    print_belief("pursuer", pursuer.belief)
    print_belief("evader", evader.belief)
    print()
    for step in range(10):
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
        print("Real actions (P,E): (%s, %s)" % (paction, eaction))
        print("   P guess E, E guess P: (%s, %s)" % (ap.eaction, ae.paction))
        print("   observation (P): %s" % op)
        print("   observation (E): %s" % oe)
        print("Updated belief (P):")
        print_belief("pursuer", pursuer.belief)
        print("Updated belief (E):")
        print_belief("evader", evader.belief)            
        print(reward)
        print("\_______________________")                

        if paction.name == "open-door"\
            or eaction.name == "open-door":
            break
        
