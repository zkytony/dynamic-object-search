from ipomdp.examples.pursuit_evasion_simple.domain import *
from ipomdp.examples.pursuit_evasion_simple.models import *
import pomdp_py
import random


pursuer_state = PState("left-door")
evader_state = PState("right-door")

# Let's do a particle filter
num_particles = 100
pursuer_particles = []
for i in range(num_particles):
    
    if random.uniform(0,1) > 0.5:
        estate = EState("left-door")
    else:
        estate = EState("right-door")

    pursuer_particles.append(JointState(pursuer_state, estate))
bp = pomdp_py.Particles(pursuer_particles)

# Let's do a belief update
Tp = TransitionModel("pursuer")
Op = ObservationModel("pursuer")
real_paction = PAction("stay")
real_eaction = PAction("move")
real_observation = Op.sample(JointState(pursuer_state, evader_state),
                             JointAction(real_paction, real_eaction))
for state in bp.particles:
    
    print(pstate)
