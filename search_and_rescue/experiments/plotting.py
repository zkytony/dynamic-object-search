import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pomdp_py import util
from search_and_rescue import ObjectObservation

def rgb2float(color):
    return (color[0]/255.0, color[1]/255.0, color[2]/255.0)

def plot_belief(ax, hist, color, cmap="Greys", size=None, zorder=1, plot_thres=0.01):
    xvals, yvals, c = [], [], []

    last_val = -1
    for state in reversed(sorted(hist, key=hist.get)):
        if last_val != -1:
            color = util.lighter(color, 1-hist[state]/last_val)
        if np.mean(np.array(color) / np.array([255, 255, 255])) < 0.99:
            tx, ty = state['pose'][:2]
            xvals.append(tx + 0.5)
            yvals.append(ty + 0.5)
            c.append(rgb2float(color))
            last_val = hist[state]
            if last_val <= 0:
                break
    if size is not None:
        size = [size] * len(xvals)
    ax.scatter(xvals, yvals, s=size, c=c, cmap=cmap, marker="s", zorder=zorder,
               vmin=0, vmax=1.0)

def plot_agent_belief(ax, agent, object_colors):
    def get_cmap(object_colors, objid):
        start_color = rgb2float(list(util.lighter(object_colors[objid], 0.7)))
        end_color = rgb2float(list(util.lighter(object_colors[objid], -0.5)))
        cmap = LinearSegmentedColormap.from_list('CustomMap',
                                                 [tuple(start_color),
                                                  tuple(end_color)])
        return cmap
    
    size = len(agent.belief.object_beliefs)*100
    zorder = 1
    for objid in agent.belief.object_beliefs:
        if objid == agent.agent_id:
            continue  # plot agent last so it appears on top.
        belief_obj = agent.belief.object_belief(objid)
        cmap = get_cmap(object_colors, objid)
        plot_belief(ax, belief_obj.get_histogram(),
                    object_colors[objid],
                    cmap=cmap, size=size, zorder=zorder)
        zorder += 1
        size = size / len(agent.belief.object_beliefs)
    # Plot the agent
    cmap = get_cmap(object_colors, agent.agent_id)
    plot_belief(ax, agent.belief.object_belief(agent.agent_id).get_histogram(),
                object_colors[agent.agent_id], cmap=cmap, size=size, zorder=zorder)

def plot_viz_observation(ax, z):
    if z is None:
        return
    xvals, yvals = [], []
    for objid in z.objposes:
        if z.for_obj(objid).pose != ObjectObservation.NULL:
            assert z.for_obj(objid).objid == objid
            lx, ly = z.for_obj(objid).pose
            xvals.append(lx+0.5)
            yvals.append(ly+0.5)
    ax.scatter(xvals, yvals, s=1000, c="yellow", zorder=0)
        
def plot_multi_agent_beliefs(agents, role_for, grid_map, object_colors,
                             viz_observations={}):
    for aid in agents:
        plt.figure(aid, figsize=(max(3, round(int(grid_map.width/3))),
                                 max(3, round(int(grid_map.length/3)))))
        plt.clf()
        fig = plt.gcf()
        ax = plt.gca()
        plot_agent_belief(ax, agents[aid], object_colors)
        if aid in viz_observations:
            plot_viz_observation(ax, viz_observations[aid])
        ax.set_title("Agent %d (%s)" % (aid, role_for(aid)),
                                        fontdict={"color": rgb2float(object_colors[aid])})
        ax.set_xlim(0, grid_map.width)
        ax.set_ylim(0, grid_map.length)
        ax.set_yticks(np.arange(0, grid_map.length+1))
        ax.set_xticks(np.arange(0, grid_map.width+1))
        ax.invert_yaxis()
        plt.grid()
        
        # Also plot obstacles (for clarity)
        oxvals = []
        oyvals = []
        for objid in grid_map.obstacles:
            x,y = grid_map.obstacles[objid]
            oxvals.append(x+0.5)
            oyvals.append(y+0.5)
            ax.scatter(oxvals, oyvals, s=500, c="black",
                       marker="x", zorder=1)
        fig.canvas.draw()
        fig.canvas.flush_events()


####### UNIT TESTS #########
worldstr=\
"""
Rx...
.x.xV
.S...    
"""
def unittest():
    from search_and_rescue.models.grid_map import GridMap
    from search_and_rescue.models.sensor import Laser2DSensor
    from search_and_rescue.env.env import unittest as env_unittest
    from search_and_rescue.env.visual import SARViz
    from search_and_rescue.agent.agent import SARAgent

    env, role_to_ids, sensors, motion_actions, look_after_move =\
        env_unittest(worldstr)
    agents = {}
    for role in role_to_ids:
        for agent_id in env.ids_for(role):
            agent = SARAgent.construct(agent_id, role, sensors[agent_id],
                                       role_to_ids, env.grid_map, motion_actions, look_after_move=look_after_move,
                                       prior={agent_id: {env.state.pose(agent_id):1.0}})
            print("agent %d own state:" % agent_id)
            print(agent.belief.mpe().object_states[agent_id])
            agents[agent_id] = agent
    viz = SARViz(env, res=30, fps=30, controller_id=None)
    plot_multi_agent_beliefs(agents, env.role_for, env.grid_map, viz.object_colors)
    plt.show()
    

if __name__ == '__main__':
    unittest()
        
