import pomdp_py
from dynamic_mos.domain.action import *
from dynamic_mos.domain.observation import *
import random
from collections import deque

class ManualPlanner(pomdp_py.Planner):
    def __init__(self, grid_map, look_after_move=False):
        self._grid_map = grid_map
        self._look_after_move = look_after_move
        self._find_next = False
        
    def update(self, agent, real_action, real_observation, **kwargs):
        robot_state = agent.belief.mpe().object_states[agent.robot_id]
        for objid in real_observation.objposes:
            if objid in robot_state["objects_found"]:
                continue  # already found this object
            observation_obj = real_observation.for_obj(objid)
            if observation_obj.pose != ObjectObservation.NULL:
                self._find_next = True
                return
        self._find_next = False
        

class RandomPlanner(ManualPlanner):
    
    # Randomly takes a valid action that is not Find.
    # Takes find when object appears in the field of view
    # in the previous look.
    def plan(self, agent):
        if self._find_next:
            self._find_next = False
            return FindAction()
        else:
            robot_pose = agent.belief.mpe().object_poses[agent.robot_id]
            valid_motions =\
                self._grid_map.valid_motions(
                    agent.robot_id,
                    robot_pose,
                    agent.policy_model.all_motion_actions)
            if self._look_after_move:
                action = random.sample(valid_motions, 1)[0]
            else:
                action = random.sample(valid_motions | set({LookAction()}), 1)[0]
            return action

        
class GreedyPlanner(ManualPlanner):
    # Greedily moves to the location of highest belief,
    # and look around. Take "Find" after seeing an object.
    def __init__(self, grid_map, look_after_move=False):
        super().__init__(grid_map, look_after_move=look_after_move)
        self._actions = deque([])

    def plan(self, agent):
        if self._find_next:
            self._find_next = False
            return FindAction()
        else:
            if len(self._actions) == 0:
                # make new path
                mpe_state = agent.belief.mpe()
                robot_state = mpe_state.object_states[agent.robot_id]
                robot_pose = robot_state.pose
                ## obtain the pose of an undetected object
                for objid in mpe_state.object_states:
                    if objid not in robot_state.objects_found:
                        object_pose = mpe_state.object_poses[objid]
                        motion_actions =\
                            self._grid_map.path_between(robot_pose, object_pose,
                                                        agent.policy_model.all_motion_actions,
                                                        return_actions=True)
                        self._actions.extend(motion_actions[:-1])  # leave room to observe
                        break
                if len(self._actions) == 0:
                    print("Warning: GreedyPlanner plan does not move the robot."\
                          "so the robot will move randomly")
                    valid_motions =\
                        self._grid_map.valid_motions(
                            agent.robot_id,
                            robot_pose,
                            agent.policy_model.all_motion_actions)
                    if self._look_after_move:
                        action = random.sample(valid_motions, 1)[0]
                    else:
                        action = random.sample(valid_motions | set({LookAction()}), 1)[0]                    
                    self._actions.append(action)
                else:
                    # Append a look action at the end
                    if not self._look_after_move:
                        self._actions.append(LookAction())

            # Return action
            return self._actions.popleft()
                    
            
        
