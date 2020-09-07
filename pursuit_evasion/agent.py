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

if __name__ == "__main__":
    pLstate = PState("left-door")
    pRstate = PState("right-door")    
    eLstate = EState("left-door")
    eRstate = EState("right-door")

    state = JointState(pLstate, eLstate)
    env = PursuitEvasionEnvironment(state)    
    
    bp0 = pomdp_py.Histogram({JointState(pLstate, eLstate):0.5,
                              JointState(pLstate, eRstate):0.5})
    be0 = pomdp_py.Histogram({JointState(pLstate, eLstate):0.5,
                              JointState(pRstate, eLstate):0.5})
    pursuer = Agent("pursuer", bp0)
    evader = Agent("evader", be0)
    print(pursuer.belief.random())

    pouct_pursuer = pomdp_py.POUCT(discount_factor=0.95,
                                   planning_time=0.4,
                                   exploration_const=100)
    pouct_evader = pomdp_py.POUCT(discount_factor=0.95,
                                   planning_time=0.4,
                                   exploration_const=100)
    print(env)
    for step in range(100):
        print("---Step %d---" % step)
        ap = pouct_pursuer.plan(pursuer)
        paction = ap.paction
        paction_value = pursuer.tree[ap].value
        op = pursuer.observation_model.sample(env.state, ap)        
        print("Pursuer:", paction, op)
        print("        ", paction_value, pouct_pursuer.last_num_sims, pursuer.belief)
        
        ae = pouct_evader.plan(evader)
        eaction = ae.eaction
        eaction_value = evader.tree[ae].value
        oe = evader.observation_model.sample(env.state, ae)
        print("Evader:", eaction, oe)
        print("       ", eaction_value, pouct_evader.last_num_sims, evader.belief)
        
        action = JointAction(paction, eaction)
        reward = env.state_transition(action, execute=True)
        print(reward)
        print(env)


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

        if paction.name == "open-door"\
            or eaction.name == "open-door":
            break
        
