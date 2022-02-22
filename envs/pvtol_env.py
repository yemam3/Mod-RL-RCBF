import numpy as np
import gym
from gym import spaces
from envs.utils import to_pixel
from rcbf_sac.utils import get_polygon_normals

class PvtolEnv(gym.Env):
    """Custom Environment that follows SafetyGym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self, obs_config='default', rand_init=False):

        super(PvtolEnv, self).__init__()

        self.dynamics_mode = 'Pvtol'
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.safe_action_space = spaces.Box(low=-2.5, high=2.5, shape=(2,))
        self.observation_space = spaces.Box(low=-1e10, high=1e10, shape=(10,))
        self.bds = np.array([[-6., -3.], [6., 3.]])

        self.dt = 0.02
        self.max_episode_steps = 1500
        self.reward_goal = 3.0
        self.goal_size = 0.3
        # Initialize Env
        self.state = None
        self.episode_step = 0
        self.initial_state = np.array([[-4.5, 2.0, 0.0, 0.0, 0.0, 1.0], [-4.5, -2.0, 0.0, 0.0, 0.0, 1.0], [4.5, 2.0, 0.0, 0.0, 0.0, 1.0]])  # x y theta v1 v2 thrust
        self.goal_pos = np.array([5.00, -1.5])
        self.rand_init = rand_init

        self.reset()

        # Build Hazards
        self.obs_config = obs_config
        if obs_config == 'default':
            self.hazards = [{'radius': 0.4, 'location': np.array([-2.5, 2.0])},
                            {'radius': 0.4, 'location': np.array([1.5, 0.2])},
                            {'radius': 0.4, 'location': np.array([-2.5, 0.])},
                            {'radius': 0.4, 'location': np.array([-.5, 1.5])},
                            {'radius': 0.4, 'location': np.array([-.5, -1.5])},
                            {'radius': 0.4, 'location': np.array([1.5, -1.8])},
                            {'radius': 0.4, 'location': np.array([-2.5, -2.2])},
                            {'radius': 0.4, 'location': np.array([5.25, 0.75])},
                            {'radius': 0.4, 'location': np.array([4.0, 0.0])}]
        elif obs_config.lower() == 'none':
            self.hazards = []
        else:
            n_hazards = 8
            hazard_radius = 0.4
            self.get_random_hazard_locations(n_hazards, hazard_radius)

        # Process Hazards for cost checking
        self.hazard_locations = np.array([hazard['location'] for hazard in self.hazards])
        self.hazards_radius = np.array([hazard['radius'] for hazard in self.hazards])
        # Viewer
        self.viewer = None


    def step(self, action):
        """Organize the observation to understand what's going on

        Parameters
        ----------
        action : ndarray
                Action that the agent takes in the environment

        Returns
        -------
        new_obs : ndarray
          The new observation with the following structure:
          [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, dist2goal]

        """

        action = np.clip(action, self.safe_action_space.low, self.safe_action_space.high)
        state, reward, done, info = self._step(action)
        return self.get_obs(), reward, done, info

    def _step(self, action):
        """

        Parameters
        ----------
        action

        Returns
        -------
        state : ndarray
            New internal state of the agent [x y theta x_d y_d thrust].
        reward : float
            Reward collected during this transition.
        done : bool
            Whether the episode terminated.
        info : dict
            Additional info relevant to the environment.
        """

        # Start with our prior for continuous time system x' = f(x) + g(x)u
        f_x = np.array([self.state[3], self.state[4], 0, -np.sin(self.state[2])*self.state[5], np.cos(self.state[2])*self.state[5]-1, 0])
        g_x = np.array([[0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [1, 0]])
        self.state += self.dt * (f_x + g_x @ action)
        # Add disturbance
        # self.state -= self.dt * 0.1 * self.get_g(self.state) @ np.array([np.cos(self.state[2]),  0])  #* np.random.multivariate_normal(self.disturb_mean, self.disturb_covar, 1).squeeze()
        self.episode_step += 1

        info = dict()

        dist_goal = self._goal_dist()
        reward = (self.last_goal_dist - dist_goal)  # -1e-3 * dist_goal
        self.last_goal_dist = dist_goal
        # Check if goal is met
        if self.goal_met():
            info['goal_met'] = True
            reward += self.reward_goal
            done = True
        else:
            done = self.episode_step >= self.max_episode_steps

        # Include constraint cost in reward
        if self.hazards and np.any(np.sum((self.state[:2] - self.hazard_locations)**2, axis=1) < self.hazards_radius**2):
            if 'cost' in info:
                info['cost'] += 0.1
            else:
                info['cost'] = 0.1
        return self.state, reward, done, info

    def goal_met(self):
        """Return true if the current goal is met this step

        Returns
        -------
        goal_met : bool
            True if the goal condition is met.

        """

        return np.linalg.norm(self.state[:2] - self.goal_pos) <= self.goal_size

    def reset(self):
        """ Reset the state of the environment to an initial state.

        Returns
        -------
        observation : ndarray
            Next observation.
        """

        self.episode_step = 0

        # Re-initialize state
        if self.rand_init:
            self.state = np.copy(self.initial_state[np.random.randint(self.initial_state.shape[0])])
        else:
            self.state = np.copy(self.initial_state[0])

        # Re-initialize last goal dist
        self.last_goal_dist = self._goal_dist()

        return self.get_obs()

    def render(self, mode='human', close=False):
        """Render the environment to the screen

        Parameters
        ----------
        mode : str
        close : bool

        Returns
        -------

        """

        if mode != 'human' and mode != 'rgb_array':
            rel_loc = self.goal_pos - self.state[:2]
            theta_error = np.arctan2(rel_loc[1], rel_loc[0]) - self.state[2]
            print('Ep_step = {}, \tState = {}, \tDist2Goal = {}, alignment_error = {}'.format(self.episode_step, self.state, self._goal_dist(), theta_error))

        screen_width = 1.5*600
        screen_height = 1.5*400

        if self.viewer is None:
            from envs import pyglet_rendering

            self.viewer = pyglet_rendering.Viewer(screen_width, screen_height)
            # Draw obstacles
            obstacles = []
            for i in range(len(self.hazards)):
                hazard_loc = self.hazards[i]['location']
                obstacles.append(pyglet_rendering.make_circle(radius=to_pixel(self.hazards[i]['radius'], shift=0), filled=True))
                obs_trans = pyglet_rendering.Transform(translation=(to_pixel(hazard_loc[0], shift=screen_width/2), to_pixel(hazard_loc[1], shift=screen_height/2)))
                obstacles[i].set_color(1.0, 0.0, 0.0)
                obstacles[i].add_attr(obs_trans)
                self.viewer.add_geom(obstacles[i])

            # Make Goal
            goal = pyglet_rendering.make_circle(radius=to_pixel(0.1, shift=0), filled=True)
            goal_trans = pyglet_rendering.Transform(translation=(to_pixel(self.goal_pos[0], shift=screen_width/2), to_pixel(self.goal_pos[1], shift=screen_height/2)))
            goal.add_attr(goal_trans)
            goal.set_color(0.0, 0.5, 0.0)
            self.viewer.add_geom(goal)

            # Make Robot
            self.robot = pyglet_rendering.make_polygon(to_pixel(np.array([[-0.3, -0.1], [0.3, -0.1], [0.3, 0.1], [-0.3, 0.1]])), filled=True)
            self.robot_trans = pyglet_rendering.Transform(translation=(to_pixel(self.state[0], shift=screen_width/2), to_pixel(self.state[1], shift=screen_height/2)))
            self.robot_trans.set_rotation(self.state[2])
            self.robot.add_attr(self.robot_trans)
            self.robot.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.robot)
            self.robot_orientation = pyglet_rendering.Line(start=(0.0, 0.0), end=(0.0, 15.0))
            self.robot_orientation.linewidth.stroke = 2
            self.robot_orientation.add_attr(self.robot_trans)
            self.robot_orientation.set_color(0, 0, 0)
            self.viewer.add_geom(self.robot_orientation)

            # Draw Boundaries
            bds = 1.5 * 37.795 * self.bds
            bds[:, 0] += screen_width/2
            bds[:, 1] += screen_height/2
            self.arena = pyglet_rendering.make_polygon(np.array([[bds[0, 0], bds[0, 1]], [bds[1, 0], bds[0, 1]], [bds[1, 0], bds[1, 1]], [bds[0, 0], bds[1, 1]]]), filled=False)
            self.viewer.add_geom(self.arena)

        if self.state is None:
            return None

        self.robot_trans.set_translation(to_pixel(self.state[0], shift=screen_width/2), to_pixel(self.state[1], shift=screen_height/2))
        self.robot_trans.set_rotation(self.state[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def get_obs(self):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------

        Returns
        -------
        observation : ndarray
          Observation: [p_x, p_y, cos(theta), sin(theta), v_x, v_y, thrust, xdir2goal, ydir2goal, exp(-dist2goal)]
        """

        rel_loc = self.goal_pos - self.state[:2]
        goal_dist = np.linalg.norm(rel_loc)
        goal_compass = self.obs_compass()  # compass to the goal

        return np.array([self.state[0], self.state[1], np.cos(self.state[2]), np.sin(self.state[2]), self.state[3], self.state[4], self.state[5], goal_compass[0], goal_compass[1], np.exp(-goal_dist)])

    def obs_compass(self):
        """
        Return a robot-centric compass observation of a list of positions.
        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.
        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """

        # Get ego vector in world frame
        vec = self.goal_pos - self.state[:2]
        # Rotate into frame
        R = np.array([[np.cos(self.state[2]), -np.sin(self.state[2])], [np.sin(self.state[2]), np.cos(self.state[2])]])
        vec = np.matmul(vec, R)
        # Normalize
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        return vec

    def _goal_dist(self):
        return np.linalg.norm(self.goal_pos - self.state[:2])

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_random_hazard_locations(self, n_hazards: int, hazard_radius: float):
        """

        Parameters
        ----------
        n_hazards : int
            Number of hazards to create
        hazard_radius : float
            Radius of hazards

        Returns
        -------
        hazards_locs : ndarray
            Numpy array of shape (n_hazards, 2) containing xy locations of hazards.
        """

        # Create buffer with boundaries
        buffered_bds = np.copy(self.bds)
        buffered_bds[0] = buffered_bds[0] + hazard_radius
        buffered_bds[1] -= hazard_radius

        hazards = []
        hazards_centers = np.zeros((n_hazards, 2))
        n = 0  # Number of hazards actually placed
        for i in range(n_hazards):
            successfully_placed = False
            iter = 0
            radius = hazard_radius * (1-0.2*2.0*(np.random.random() - 0.5))
            while not successfully_placed and iter < 100:
                hazards_centers[n] = (buffered_bds[1] - buffered_bds[0]) * np.random.random(2) + buffered_bds[0]
                successfully_placed = np.all(np.linalg.norm(hazards_centers[:n] - hazards_centers[[n]], axis=1) > 3.5*hazard_radius)
                successfully_placed = np.logical_and(successfully_placed, np.linalg.norm(self.goal_pos - hazards_centers[n]) > 3.0*hazard_radius)
                successfully_placed = np.logical_and(successfully_placed, np.all(np.linalg.norm(self.initial_state[:, :2] - hazards_centers[[n]], axis=1) > 3.0*hazard_radius))
                iter += 1
            if not successfully_placed:
                continue
            hazards.append({'location': hazards_centers[n], 'radius': radius})
            n += 1

        self.hazards = hazards


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import torch
    from rcbf_sac.utils import to_tensor, to_numpy
    from rcbf_sac.cbf_qp import CascadeCBFLayer
    from rcbf_sac.diff_cbf_qp import CBFQPLayer
    from rcbf_sac.dynamics import DynamicsModel
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="SafetyGym", help='Either SafetyGym or Unicycle.')
    parser.add_argument('--gp_model_size', default=2000, type=int, help='gp')
    parser.add_argument('--k_d', default=3.0, type=float)
    parser.add_argument('--gamma_b', default=50, type=float)
    parser.add_argument('--l_p', default=0.03, type=float, help="Look-ahead distance for unicycle dynamics output.")
    parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')
    args = parser.parse_args()

    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    env = PvtolEnv()
    dynamics_model = DynamicsModel(env, args)
    cbf_wrapper = CBFQPLayer(env, args, gamma_b=args.gamma_b, k_d=args.k_d)

    obs = env.reset()
    done = False
    episode_reward = 0
    episode_step = 0

    while not done:
        # Take Action and get next state
        # random_action = env.action_space.sample()
        state = dynamics_model.get_state(obs)
        #random_action = env.action_space.sample()
        random_action = np.array([1.0-state[5], 0.2*(-np.pi/4 - state[2])])
        disturb_mean, disturb_std = dynamics_model.predict_disturbance(state)
        state = to_tensor(state, torch.FloatTensor, 'cpu')
        random_action = to_tensor(random_action, torch.FloatTensor, 'cpu')
        disturb_mean = to_tensor(disturb_mean, torch.FloatTensor, 'cpu')
        disturb_std = to_tensor(disturb_std, torch.FloatTensor, 'cpu')
        random_action = to_numpy(cbf_wrapper.get_safe_action(state, random_action, disturb_mean, disturb_std))
        obs, reward, done, info = env.step(random_action)
        env.render()
        plt.pause(0.01)
        episode_reward += reward
        episode_step += 1
        print('step {} \tepisode_reward = {}'.format(episode_step, episode_reward))
    plt.show()

