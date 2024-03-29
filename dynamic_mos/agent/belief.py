# Defines the belief distribution and update for the 2D Multi-Object Search domain;
#
# The belief distribution is represented as a Histogram (or Tabular representation).
# Since the observation only contains mapping from object id to their location,
# the belief update has no leverage on the shape of the sensing region; this is
# makes the belief update algorithm more general to any sensing region but then
# requires updating the belief by iterating over the state space in a nested
# loop. The alternative is to use particle representation but also object-oriented.
# We try both here.
#
# We can directly make use of the Histogram and Particle classes in pomdp_py.
import pomdp_py
import random
import copy
from ..domain.state import *

class MosOOBelief(pomdp_py.OOBelief):
    """This is needed to make sure the belief is sampling the right
    type of State for this problem."""
    def __init__(self, robot_id, object_beliefs):
        """
        robot_id (int): The id of the robot that has this belief.
        object_beliefs (objid -> GenerativeDistribution)
            (includes robot)
        """
        self.robot_id = robot_id
        super().__init__(object_beliefs)

    def mpe(self, **kwargs):
        return MosOOState(self.robot_id,
                        pomdp_py.OOBelief.mpe(self, **kwargs).object_states)

    def random(self, **kwargs):
        return MosOOState(self.robot_id,
                          pomdp_py.OOBelief.random(self, **kwargs).object_states)


### Belief Update ###
def belief_update(agent, real_action, real_observation,
                  next_robot_state, robot_state,
                  planner, dynamic_object_ids=set({})):
    """Updates the agent's belief; The belief update may happen
    through planner update (e.g. when planner is POMCP)."""
    # Updates the planner; In case of POMCP, agent's belief is also updated.
    planner.update(agent, real_action, real_observation)

    # Update agent's belief, when planner is not POMCP
    if not isinstance(planner, pomdp_py.POMCP):
        # Update belief for every undetected object
        for objid in agent.cur_belief.object_beliefs:
            if objid in next_robot_state['objects_found']:
                continue  # already found this object
            belief_obj = agent.cur_belief.object_belief(objid)

            # Histogram belief update
            if isinstance(belief_obj, pomdp_py.Histogram):
                if objid == agent.robot_id:
                    # Assuming the agent can observe its own state:
                    new_belief = pomdp_py.Histogram({next_robot_state: 1.0})
                else:
                    # This is doing
                    #    B(si') = normalizer * O(oi|si',sr',a) * sum_s T(si'|s,a)*B(si)
                    static_transition = (objid != agent.robot_id) and (objid not in dynamic_object_ids)

                    # The following sets up a state space where the time step have advanced.
                    next_state_space = None
                    if not static_transition:
                        next_state_space = set({})
                        for state in belief_obj:
                            next_state = copy.deepcopy(state)
                            next_state["time"] = state["time"] + 1
                            next_state_space.add(next_state)
                    
                    new_belief = pomdp_py.update_histogram_belief(
                        belief_obj, real_action,
                        real_observation.for_obj(objid),
                        agent.observation_model[objid],
                        agent.transition_model[objid],
                        # The agent knows the objects are static.
                        static_transition=static_transition,
                        next_state_space=next_state_space,
                        oargs={"next_robot_state": next_robot_state},
                        targs={"robot_state": robot_state})

            # Weighted particles belief update
            elif isinstance(belief_obj, pomdp_py.WeightedParticles):
                if objid == agent.robot_id:
                    # Assuming the agent can observe its own state:                    
                    new_belief = pomdp_py.Histogram({next_robot_state: 1.0})
                else:
                    new_belief = pomdp_py.update_weighted_particles_belief(
                        belief_obj, real_action, real_observation.for_obj(objid),
                        agent.observation_model[objid],
                        agent.transition_model[objid],
                        oargs={"next_robot_state": next_robot_state},
                        targs={"robot_state": robot_state})
                    
            else:
                raise ValueError("Unexpected program state.")

            agent.cur_belief.set_object_belief(objid, new_belief)    


def initialize_belief(agent, dim, robot_id, object_ids, prior={},
                      representation="histogram", robot_orientations={},
                      num_particles=100, grid_map=None):
    """
    Returns a GenerativeDistribution that is the belief representation for
    the multi-object search problem.

    Args:
        dim (tuple): a tuple (width, length) of the search space gridworld.
        robot_id (int): robot id that this belief is initialized for.
        object_ids (dict): a set of object ids that we want to model the belief distribution
                          over; They are `assumed` to be the target objects, not obstacles,
                          because the robot doesn't really care about obstacle locations and
                          modeling them just adds computation cost.
        prior (dict): A mapping {(objid|robot_id) -> {(x,y) -> [0,1]}}. If used, then 
                      all locations not included in the prior will be treated to have 0 probability.
                      If unspecified for an object, then the belief over that object is assumed
                      to be a uniform distribution.
        robot_orientations (dict): Mapping from robot id to their initial orientation (radian).
                                   Assumed to be 0 if robot id not in this dictionary.
        num_particles (int): Maximum number of particles used to represent the belief
        grid_map (GridMap): The occupancy grid map the agent equips. By default None (i.e.
                            agent isn't equipped with a map).

    Returns:
        GenerativeDistribution: the initial belief representation.
    """
    if agent.cur_belief is not None:
        raise ValueError("Agent already has initial belief!")
    if representation == "histogram":
        return _initialize_histogram_belief(agent, dim, robot_id, object_ids,
                                            prior, robot_orientations,
                                            grid_map=grid_map)
    elif representation == "particles":
        return _initialize_particles_belief(agent, dim, robot_id, object_ids,
                                            robot_orientations, num_particles=num_particles,
                                            grid_map=grid_map)
    elif representation == "weighted_particles":
        return _initialize_weighted_particles_belief(agent, dim, robot_id, object_ids,
                                                     prior, robot_orientations,
                                                     num_particles=num_particles,
                                                     grid_map=grid_map) 
    else:
        raise ValueError("Unsupported belief representation %s" % representation)

    
def _initialize_histogram_belief(agent, dim, robot_id, object_ids,
                                 prior, robot_orientations,
                                 grid_map=None):
    """
    Returns the belief distribution represented as a histogram
    """
    oo_hists = {}  # objid -> Histogram
    width, length = dim
    for objid in object_ids:
        hist = {}  # pose -> prob
        total_prob = 0
        dynamic_object = agent.motion_policy(objid) is not None
        time = 0 if dynamic_object else -1
        if objid in prior:
            # prior knowledge provided. Just use the prior knowledge
            for pose in prior[objid]:
                # We will assume in the informed case, the dynamic object
                # always starts from the 0th location in its path.
                state = ObjectState(objid, "target", pose, time=time)
                hist[state] = prior[objid][pose]
                total_prob += hist[state]

        for x in range(width):
            for y in range(length):
                if grid_map is not None\
                   and (x,y) in grid_map.obstacle_poses:
                    # belief here is always going to be zero.
                    # so exclude it from the histogram
                    continue

                # If prior knowledge provided, just set a probability of
                # 0 to states not included in the prior. Otherwise,
                # we are dealing with uniform prior.
                state = ObjectState(objid, "target", (x,y), time=time)
                if objid in prior:
                    hist[state] = 1e-9 # almost zero probability
                else:
                    hist[state] = 1.0  # uniform prior
                    total_prob += hist[state]
                    
        # Normalize
        for state in hist:
            hist[state] /= total_prob

        hist_belief = pomdp_py.Histogram(hist)
        oo_hists[objid] = hist_belief

    # For the robot, we assume it can observe its own state;
    # Its pose must have been provided in the `prior`.
    assert robot_id in prior, "Missing initial robot pose in prior."
    init_robot_pose = list(prior[robot_id].keys())[0]
    oo_hists[robot_id] =\
        pomdp_py.Histogram({RobotState(robot_id, init_robot_pose, (), None): 1.0})
        
    return MosOOBelief(robot_id, oo_hists)


# WARNING: OUT OF DATE.
def _initialize_particles_belief(agent, dim, robot_id, object_ids, prior,
                                 robot_orientations, num_particles=100,
                                 grid_map=None):
    """This returns a single set of particles that represent the distribution over a
    joint state space of all objects.

    Since it is very difficult to provide a prior knowledge over the joint state
    space when the number of objects scales, the prior (which is
    object-oriented), is used to create particles separately for each object to
    satisfy the prior; That is, particles beliefs are generated for each object
    as if object_oriented=True. Then, `num_particles` number of particles with
    joint state is sampled randomly from these particle beliefs.

    """
    # For the robot, we assume it can observe its own state;
    # Its pose must have been provided in the `prior`.
    assert robot_id in prior, "Missing initial robot pose in prior."
    init_robot_pose = list(prior[robot_id].keys())[0]
    
    oo_particles = {}  # objid -> Particageles
    width, length = dim
    for objid in object_ids:
        particles = [RobotState(robot_id, init_robot_pose, (), None)]  # list of states; Starting the observable robot state.
        if objid in prior:
            # prior knowledge provided. Just use the prior knowledge
            prior_sum = sum(prior[objid][pose] for pose in prior[objid])
            for pose in prior[objid]:
                state = ObjectState(objid, "target", pose)
                amount_to_add = (prior[objid][pose] / prior_sum) * num_particles
                for _ in range(amount_to_add):
                    particles.append(state)
        else:
            # no prior knowledge. So uniformly sample `num_particles` number of states.
            for _ in range(num_particles):
                x = random.randrange(0, width)
                y = random.randrange(0, length)
                if grid_map is not None\
                   and (x,y) in grid_map.obstacle_poses:
                    # belief here is always going to be zero.
                    # so exclude it from the histogram
                    continue                
                state = ObjectState(objid, "target", (x,y))
                particles.append(state)

        particles_belief = pomdp_py.Particles(particles)
        oo_particles[objid] = particles_belief
        
    # Return Particles distribution which contains particles
    # that represent joint object states
    particles = []
    for _ in range(num_particles):
        object_states = {}
        for objid in oo_particles:
            random_particle = random.sample(oo_particles[objid], 1)[0]
            object_states[_id] = copy.deepcopy(random_particle)
        particles.append(MosOOState(robot_id, object_states))
    return pomdp_py.Particles(particles)


def _initialize_weighted_particles_belief(agent, dim, robot_id,
                                          object_ids, prior, robot_orientations, num_particles=100,
                                          grid_map=None):
    oo_particles = {}  # objid -> WeightedParticles
    width, length = dim
    for objid in object_ids:
        particle_candidates = []  # weighted particles ((state, weight))

        if objid in prior:
            total_prob = sum([prior[objid][pose] for pose in prior])
            for pose in prior[objid]:
                particle = ObjectState(objid, "target", pose)
                weight = prior[objid][pose] / total_prob
                particle_candidates.append((particle, weight))

        else:
            # No prior knowledge
            for x in range(width):
                for y in range(length):
                    if grid_map is not None\
                       and (x,y) in grid_map.obstacle_poses:
                        continue
                        
                    particle = ObjectState(objid, "target", (x,y))
                    particle_candidates.append((particle, 1.0 / (width * length)))

        # Sample randomly `num_particles` number of particles from the list
        particles = []
        for _ in range(num_particles):
            particles.append(random.choice(particle_candidates))
        oo_particles[objid] = pomdp_py.WeightedParticles(particles)

    # For the robot, assume it can observe its own state.
    # Its pose must have been provided in the `prior`.
    assert robot_id in prior, "Missing initial robot pose in prior."
    init_robot_pose = list(prior[robot_id].keys())[0]
    oo_particles[robot_id] = pomdp_py.Histogram({
        RobotState(robot_id, init_robot_pose, (), None): 1.0})
    return MosOOBelief(robot_id, oo_particles)
    

"""If `object oriented` is True, then just like histograms, there will be
one set of particles per object; Otherwise, there is a single set
of particles that represent the distribution over a joint state space
of all <objects.

When updating the particle belief, Monte Carlo simulation is used instead of
computing the probabilities using T/O models. This means one must sample
(s',o,r) from G(s,a). If this belief representation if object oriented, then
you have N particle sets for N objects. Thanks to the fact that in this
particular domain, objects are static, you could have si' = si if i is an
object. However, if robot state sr' needs to consider collision with other
objects, then it can't be obtained just from sr. This means eventually you
would have to build an s by sampling randomly from the particle set for each
object.

More details on the non-object-oriented case: Since it is extremely
difficult to provide a prior knowledge over the joint state space when
the number of objects scales, the prior (which is object-oriented),
is used to create particles separately for each object to satisfy
the prior; That is, particles beliefs are generated for each object
as if object_oriented=True. Then, `num_particles` number of particles
with joint state is sampled randomly from these particle beliefs.
"""
