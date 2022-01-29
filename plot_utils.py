from rcbf_sac.utils import to_tensor, to_numpy
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from rcbf_sac.utils import prYellow
import os

def plot_value_function(env, agent, dynamics_model, save_path='', safe_action=False):

    fn = save_path + '/figures/vf_plot.png'
    if save_path and os.path.exists(fn):
        prYellow('Not plotting value function since figure already exists at {}'.format(fn))
        return

    if env.dynamics_mode == 'Unicycle':
        res = 20
        xs = np.linspace(env.bds[0][0], env.bds[1][0], res)
        ys = np.linspace(env.bds[0][1], env.bds[1][1], res)
        ths = np.linspace(-np.pi, np.pi, res)
        xxs, yys, thhs = np.meshgrid(xs, ys, ths)
        states = np.vstack((xxs.ravel(), yys.ravel(), thhs.ravel())).transpose()  # N x 3
        actions = np.zeros((states.shape[0], env.action_space.shape[0]))
        obs = np.zeros((states.shape[0], env.observation_space.shape[0]))
        # Get Actions and Observations corresponding to Each State (TODO:can be vectorized if env is vectorized...)
        for i in tqdm(range(states.shape[0])):
            env.state = states[i]
            obs[i] = env.get_obs()
            actions[i], _ = agent.select_action(obs[i], dynamics_model, evaluate=True, safe_action=safe_action)
        obs = to_tensor(obs, torch.FloatTensor, agent.device)
        actions = to_tensor(actions, torch.FloatTensor, agent.device)
        vf1, vf2 = agent.critic(obs, actions)  # Each is Nx3
        vf1 = to_numpy(vf1).squeeze()
        vf2 = to_numpy(vf2).squeeze()
        vf = np.min(np.vstack((vf1, vf2)).transpose(), axis=1, keepdims=True)
        # Take Max over thetas
        vf = np.max(vf.reshape((res, res, res)), axis=2, keepdims=False)
        fig, ax = plt.subplots()

        # Add obstacles, goal and initial pos
        for i in range(len(env.hazards)):
            if env.hazards[i]['type'] == 'circle':
                ax.add_patch(plt.Circle(env.hazards[i]['location'], env.hazards[i]['radius'], color='r', fill=False, linewidth=2.0))
            elif env.hazards[i]['type'] == 'polygon':
                ax.add_patch(plt.Polygon(env.hazards[i]['vertices'], color='r', fill=False, linewidth=2.0))

        ax.add_patch(plt.Circle(env.goal_pos, env.goal_size, color='g', fill=False, linewidth=2.0))
        for initial_state in env.initial_state:
            ax.add_patch(plt.Circle(initial_state[:2], 0.3, color='b', fill=False, linewidth=2.0))

        c = ax.pcolormesh(xs, ys, vf.reshape((res, res)), cmap='hot', shading='nearest', vmin=1.0, vmax=2.0, alpha=0.8)
        fig.colorbar(c, ax=ax)
        if save_path:
            if not os.path.exists(save_path + '/figures'):
                os.mkdir(save_path + '/figures')
            plt.savefig(save_path + '/figures/vf_plot.png')
        else:
            plt.show()
        plt.close()