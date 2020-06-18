import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pomdp_py import util

def plot_belief(ax, belief, cmap="Greys", size=None, zorder=1):
    xvals, yvals, c = [], [], []
    for state in belief:
        x, y = state.pose[:2]
        xvals.append(x+0.5)  # to show it between grid lines
        yvals.append(y+0.5)
        c.append(belief[state])
    if size is not None:
        size = [size] * len(xvals)
    ax.scatter(xvals, yvals, s=size, c=c, cmap=cmap, marker="s", zorder=zorder,
               vmin=0, vmax=1.0)

def plot_agent_belief(ax, agent, object_colors):
    def get_cmap(object_colors, objid):
        start_color = list(util.lighter(object_colors[objid], 0.9))
        end_color = list(object_colors[objid])
        for i in range(len(start_color)):
            start_color[i] /= 255.0
            end_color[i] /= 255.0
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
        plot_belief(ax, belief_obj, cmap=cmap, size=size, zorder=zorder)
        zorder += 1
        size = size / len(agent.belief.object_beliefs)
    # Plot the agent
    cmap = get_cmap(object_colors, agent.agent_id)
    plot_belief(ax, agent.belief.object_belief(agent.agent_id),
                cmap=cmap, size=size, zorder=zorder)
    
        
def plot_multi_agent_beliefs(agents, role_for, grid_map, object_colors):
    for aid in agents:
        plt.figure(aid)
        plt.clf()
        fig = plt.gcf()
        ax = plt.gca()
        plot_agent_belief(ax, agents[aid], object_colors)
        ax.set_title("Agent %d (%s)" % (aid, role_for(aid)))
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
            ax.scatter(oxvals, oyvals, s=1000, c="black",
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
        
