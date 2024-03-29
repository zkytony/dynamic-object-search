import pygame
import pomdp_py
from pomdp_py import util
import cv2
import math
import numpy as np
import random
import copy
import time
from search_and_rescue.env.env import *
from search_and_rescue.env.action import *
from dynamic_mos.experiments.world_types import *
from dynamic_mos.example_worlds import place_objects

# Deterministic way to get object color
def object_color(objid, count):
    color = [107, 107, 107]
    if count % 5 == 0:
        color[0] += random.randint(120, 144)
        color[0] = max(12, min(222, color[0]))
    elif count % 5 == 1:
        color[1] += random.randint(120, 144)        
        color[1] = max(12, min(222, color[1]))        
    elif count % 5 == 2:
        color[2] += random.randint(120, 144)        
        color[2] = max(12, min(222, color[2]))
    elif count % 5 == 3:
        color[0] = random.randint(120, 144)        
        color[0] = max(12, min(222, color[0]))        
        color[1] = random.randint(120, 144)        
        color[1] = max(12, min(222, color[1]))
    return tuple(color)


class SARViz:
    
    def __init__(self, env,
                 res=30, fps=30, controller_id=None, game_mode=False):
        """
        If game mode is True, then will not show the environment image.
        Instead, records the environment state. After game is over,
        review the environment states by drawing them.
        """
        self._env = env

        self._res = res
        self._img = self._make_gridworld_image(res)
        
        self._controller_id = controller_id  # the id that is controlled by man.
        self._running = False
        self._fps = fps
        self._playtime = 0.0
        self._game_mode = game_mode

        # Things to dynamically visualize/update
        self._last_observation = {}  # map from robot id to OOObservation
        self._last_viz_observation = {}  # map from robot id to OOObservation        
        self._last_action = {}  # map from robot id to Action
        self._img_history = []

        # Generate some colors, one per object
        colors = {}
        random.seed(1)
        for i, agent_id in enumerate(sorted(env.ids_for("searcher"))):
            colors[agent_id] = object_color(agent_id, 0)
            print("Robot %d is assigned color %s" % (agent_id, colors[agent_id]))            
        for i, agent_id in enumerate(sorted(env.ids_for("victim"))):
            colors[agent_id] = object_color(agent_id, 1)
            print("Object %d is assigned color %s" % (agent_id, colors[agent_id]))
        for i, agent_id in enumerate(sorted(env.ids_for("suspect"))):
            colors[agent_id] = object_color(agent_id, 2)
            print("Robot %d is assigned color %s" % (agent_id, colors[agent_id]))
        for i, agent_id in enumerate(sorted(env.ids_for("target"))):
            colors[agent_id] = object_color(agent_id, 3)
            print("Robot %d is assigned color %s" % (agent_id, colors[agent_id]))            
        random.seed()
        self._colors = colors

    @property
    def object_colors(self):
        return self._colors

    def _make_gridworld_image(self, r):
        # Preparing 2d array
        w, l = self._env.grid_map.width, self._env.grid_map.length
        arr2d = np.full((w, l), 0)
        state = self._env.state
        for objid in self._env.grid_map.obstacles:
            pose = state.object_states[objid]["pose"]
            arr2d[pose[0], pose[1]] = 1  # obstacle

        # Creating image
        img = np.full((w*r,l*r,3), 255, dtype=np.int32)
        for x in range(w):
            for y in range(l):
                if arr2d[x,y] == 0:    # free
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (255, 255, 255), -1)
                elif arr2d[x,y] == 1:  # obstacle
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (40, 31, 3), -1)
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0), 1, 8)                    
        return img
    
    @property
    def img_width(self):
        return self._img.shape[0]
    
    @property
    def img_height(self):
        return self._img.shape[1]

    @property
    def last_observation(self):
        return self._last_observation

    @property
    def controller_id(self):
        return self._controller_id
    
    def update(self, agent_id, action, observation, viz_observation):
        """
        Update the visualization after there is new real action and observation
        and updated belief.

        Args:
            observation (MosOOObservation): Real observation
            viz_observation (MosOOObservation): An observation used to visualize
                                                the sensing region.
        """
        self._last_action[agent_id] = action
        self._last_observation[agent_id] = observation
        self._last_viz_observation[agent_id] = viz_observation

    def record_game_state(self, img):
        self._img_history.append(copy.deepcopy(img))

    def replay(self, interval=0.5):
        for img in self._img_history:
            pygame.surfarray.blit_array(self._display_surf, img)
            pygame.display.flip()
            time.sleep(interval)

    @property
    def img_history(self):
        return self._img_history
        
    @staticmethod
    def draw_agent(img, x, y, th, size, color=(255,12,12)):
        radius = int(round(size / 2))
        cv2.circle(img, (y+radius, x+radius), radius, color, thickness=2)

        endpoint = (y+radius + int(round(radius*math.sin(th))),
                    x+radius + int(round(radius*math.cos(th))))
        cv2.line(img, (y+radius,x+radius), endpoint, color, 2)

    @staticmethod
    def draw_observation(img, z, rx, ry, rth, r, size, color=(12,12,255), border_color=None):
        assert type(z) == JointObservation, "%s != JointObservation" % (str(type(z)))
        radius = int(round(r / 2))
        for objid in z.objposes:
            if z.for_obj(objid).pose != ObjectObservation.NULL:
                lx, ly = z.for_obj(objid).pose
                if border_color is not None:
                    cv2.circle(img, (ly*r+radius,
                                     lx*r+radius), size+2, border_color, thickness=-1)
                cv2.circle(img, (ly*r+radius,
                                 lx*r+radius), size, color, thickness=-1)

    # PyGame interface functions
    def on_init(self):
        """pygame init"""
        pygame.init()  # calls pygame.font.init()
        # init main screen and background
        self._display_surf = pygame.display.set_mode((self.img_width,
                                                      self.img_height),
                                                     pygame.HWSURFACE)
        self._background = pygame.Surface(self._display_surf.get_size()).convert()
        self._clock = pygame.time.Clock()

        # Font
        self._myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self._running = True

    def on_render(self, display=True):
        # self._display_surf.blit(self._background, (0, 0))
        img = self.render_env(self._display_surf, display=display)
        caption = "FPS: {0:.2f}".format(self._clock.get_fps())
        caption += " | " + self.ctrl_status()
        pygame.display.set_caption(caption)
        pygame.display.flip()
        return img

    def ctrl_status(self):
        status = ""
        if self._controller_id is not None:
            status += "controlling: %d (%s)" % (self._controller_id,
                                                    self._env.role_for(self._controller_id))
            # last action
            last_action = self._last_action.get(self._controller_id, None)
            last_action_str = "no_action" if last_action is None else str(last_action)
            status += " | %s" % last_action_str

            # set
            state_controlled = self._env.state.object_states[self._controller_id]
            if hasattr(state_controlled, "fov_objects"):
                status += " | %s" % str(state_controlled.fov_objects)
            elif hasattr(state_controlled, "objects_found"):
                status += " | %s" % str(state_controlled.objects_found)
        return status

    def render_single_agent(self, img, objid, pose, res):
        rx, ry, rth = pose
        r = res  # just shorter for resolution.
        last_observation = self._last_observation.get(objid, None)
        last_viz_observation = self._last_viz_observation.get(objid, None)
        if last_viz_observation is not None:
            # Observation that covers the whole fov
            color = (200, 200, 12)
            SARViz.draw_observation(img, last_viz_observation,
                                    rx, ry, rth, r, r//4, color=color,
                                    border_color=self._colors[objid])
        if last_observation is not None:
            # Observation that correspond to detected objects
            color = (20, 20, 180)
            SARViz.draw_observation(img, last_observation,
                                    rx, ry, rth, r, r//8, color=color)
        SARViz.draw_agent(img, rx*r, ry*r, rth, r, color=self._colors[objid])

    def render_single_target(self, img, objid, pose, res):
        x, y = pose
        r = res
        color = util.lighter(self._colors[objid], -0.3)
        cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r), color, -1)

    def render_objects(self, img, res):
        """Render all objects (including agents)"""
        for objid in self._env.state.object_states:
            if self._env.role_for(objid) in {"searcher", "victim", "suspect"}:
                self.render_single_agent(img, objid, self._env.state.pose(objid), res)
            elif self._env.role_for(objid) == "obstacle":
                continue # already drawn obstacles
            elif self._env.role_for(objid) == "target":
                self.render_single_target(img, objid, self._env.state.pose(objid), res)

    def render_env(self, display_surf, display=True):
        # draw robot, a circle and a vector
        img = np.copy(self._img)
        self.render_objects(img, self._res)
        if (not self._game_mode) and display:
            pygame.surfarray.blit_array(display_surf, img)
        return img

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            u = None  # control signal according to motion model
            action = None  # control input by user

            # euclidean axis model
            if event.key == pygame.K_a:
                action = MoveWest
            elif event.key == pygame.K_d:
                action = MoveEast
            elif event.key == pygame.K_x:
                action = MoveSouth
            elif event.key == pygame.K_w:
                action = MoveNorth
            elif event.key == pygame.K_SPACE:
                action = Look
            elif event.key == pygame.K_s:
                action = Stay     
            elif event.key == pygame.K_RETURN:
                action = Find
                
            if (self._controller_id is None) or (action is None):
                return
            else:
                robot_id = self._controller_id
            if action is None:
                return
            return action

    def wait_for_action(self, interval=0.5):
        action = None
        printed = False
        while action is None:
            for event in pygame.event.get():
                action = self.on_event(event)
            if action is None:
                time.sleep(interval)
                if not printed:
                    print("Please input action")
                    printed = True
        return action

    def on_loop(self):
        self._playtime += self._clock.tick(self._fps) / 1000.0
        
    def on_cleanup(self):
        pygame.quit()
 
    def on_execute(self, action_callback):
        if self.on_init() == False:
            self._running = False
 
        while( self._running ):
            for event in pygame.event.get():
                action = self.on_event(event)
                if action is not None:
                    action_callback(self, self._env, action)
            self.on_loop()
            self.on_render()
        self.on_cleanup()


####### UNIT TESTS ######
worldstr=\
"""
Rx...
.x.xV
.S...    
"""
def action_callback(viz, env, action):
    # State transition
    rewards = env.state_transition(ActionCollection({viz.controller_id: action}),
                                                    execute=True)
    print(env.state.object_states[viz.controller_id])    
    # Sample observation
    z = {}      # shows only relevant object
    for objid in env.state.object_states:
        if objid == viz.controller_id:
            continue
        z[objid] = env.sensors[viz.controller_id].random(env.state,
                                                         action,
                                                         object_id=objid).pose
    observation = JointObservation(z)  # z is objposes
    print(observation)

    # Update visualization
    viz_state = {}
    z_viz = {}  # shows the whole FOV
    for x in range(env.grid_map.width):
        for y in range(env.grid_map.length):
            viz_objid = len(viz_state)
            viz_state[viz_objid] = ObstacleState(viz_objid, (x,y))
    viz_state[viz.controller_id] = copy.deepcopy(env.state.object_states[viz.controller_id])
    viz_state = JointState(viz_state)
    for objid in viz_state.object_states:
        z_viz[objid] = env.sensors[viz.controller_id].random(viz_state,
                                                             action,
                                                             object_id=objid).pose
    observation_fov = JointObservation(z_viz)
    viz.update(viz.controller_id, action, observation, observation_fov, None)
    print(viz.ctrl_status())    
    

def unittest(worldstr):
    from search_and_rescue.models.grid_map import GridMap
    from search_and_rescue.env.action import create_motion_actions

    laserstr = make_laser_sensor(90, (1, 3), 0.5, False)
    worldstr = equip_sensors(worldstr, {"S": laserstr,
                                        "V": laserstr,
                                        "R": laserstr})
    dims, robots, objects, obstacles, sensors, role_to_ids = interpret(worldstr)
    grid_map = GridMap(dims[0], dims[1],
                       {objid: objects[objid].pose
                        for objid in obstacles})
    motion_actions = create_motion_actions(can_stay=True)
    env = SAREnvironment.construct(role_to_ids,
                                   {**robots, **objects},
                                   grid_map, motion_actions, sensors,
                                   look_after_move=True)
    viz = SARViz(env, res=30, fps=30, controller_id=7000)
    print(env.state.object_states[viz.controller_id])    
    viz.on_execute(action_callback)    

if __name__ == '__main__':
    # mapstr, free_locations = create_free_world(6,6)#create_connected_hallway_world(9, 1, 1, 3, 3) # #create_two_room_loop_world(5,5,3,1,1)#create_two_room_world(4,4,3,1)
    random.seed(100)
    mapstr, free_locations = create_hallway_world(9, 2, 1, 3, 3)    
    searcher_pose = random.sample(free_locations, 1)[0]
    victim_pose = random.sample(free_locations - {searcher_pose}, 1)[0]
    suspect_pose = random.sample(free_locations - {victim_pose, searcher_pose}, 1)[0]

    mapstr = place_objects(mapstr,
                           {"R": searcher_pose,
                            "T": victim_pose,
                            "P": suspect_pose})
    unittest(mapstr)
