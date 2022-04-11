import argparse
import numpy as np
import torch
from rcbf_sac.dynamics import DYNAMICS_MODE
from rcbf_sac.utils import to_tensor, to_numpy, prRed, get_polygon_normals, sort_vertices_cclockwise
from time import time
from qpth.qp import QPFunction

class CBFQPLayer:

    def __init__(self, env, args, gamma_b=100, k_d=1.5, l_p=0.03):
        """Constructor of CBFLayer.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        gamma_b : float, optional
            gamma of control barrier certificate.
        k_d : float, optional
            confidence parameter desired (2.0 corresponds to ~95% for example).
        """

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.env = env
        self.u_min, self.u_max = self.get_control_bounds()
        self.gamma_b = gamma_b

        if self.env.dynamics_mode not in DYNAMICS_MODE:
            raise Exception('Dynamics mode not supported.')

        if self.env.dynamics_mode == 'Unicycle':
            self.k_d = k_d
            self.l_p = l_p
        # elif self.env.dynamics_mode == 'SimulatedCars':
            # self.num_cbfs = 2

        self.action_dim = env.action_space.shape[0]
        # self.num_ineq_constraints = self.num_cbfs + 2 * self.action_dim

    def get_safe_action(self, state_batch, action_batch, mean_pred_batch, sigma_batch, modular=False):
        """

        Parameters
        ----------
        state_batch : torch.tensor or ndarray
        action_batch : torch.tensor or ndarray
            State batch
        mean_pred_batch : torch.tensor or ndarray
            Mean of disturbance
        sigma_batch : torch.tensor or ndarray
            Standard deviation of disturbance

        Returns
        -------
        final_action_batch : torch.tensor
            Safe actions to take in the environment.
        """

        # batch form if only a single data point is passed
        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            action_batch = action_batch.unsqueeze(0)
            state_batch = state_batch.unsqueeze(0)
            mean_pred_batch = mean_pred_batch.unsqueeze(0)
            sigma_batch = sigma_batch.unsqueeze(0)

        if modular and self.env.dynamics_mode != 'Pvtol':
            final_action = torch.clamp(action_batch, self.u_min.repeat(action_batch.shape[0], 1), self.u_max.repeat(action_batch.shape[0], 1))
        else:
            start_time = time()
            Ps, qs, Gs, hs = self.get_cbf_qp_constraints(state_batch, action_batch, mean_pred_batch, sigma_batch, modular=modular)
            build_qp_time = time()
            safe_action_batch = self.solve_qp(Ps, qs, Gs, hs)
            # prCyan('Time to get constraints = {} - Time to solve QP = {} - time per qp = {} - batch_size = {} - device = {}'.format(build_qp_time - start_time, time() - build_qp_time, (time() - build_qp_time) / safe_action_batch.shape[0], Ps.shape[0], Ps.device))
            # The actual safe action is the cbf action + the nominal action
            final_action = torch.clamp(action_batch + safe_action_batch, self.u_min.repeat(action_batch.shape[0], 1), self.u_max.repeat(action_batch.shape[0], 1))

        return final_action if not expand_dims else final_action.squeeze(0)

    def solve_qp(self, Ps: torch.Tensor, qs: torch.Tensor, Gs: torch.Tensor, hs: torch.Tensor):
        """Solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Parameters
        ----------
        Ps : torch.Tensor
            (batch_size, n_u+1, n_u+1)
        qs : torch.Tensor
            (batch_size, n_u+1)
        Gs : torch.Tensor
            (batch_size, num_ineq_constraints, n_u+1)
        hs : torch.Tensor
            (batch_size, num_ineq_constraints)
        Returns
        -------
        safe_action_batch : torch.tensor
            The solution of the qp without the last dimension (the slack).
        """


        Ghs = torch.cat((Gs, hs.unsqueeze(2)), -1)
        Ghs_norm = torch.max(torch.abs(Ghs), dim=2, keepdim=True)[0]
        Gs /= Ghs_norm
        hs = hs / Ghs_norm.squeeze(-1)
        sol = self.cbf_layer(Ps, qs, Gs, hs, solver_args={"check_Q_spd": False, "maxIter": 100000, "notImprovedLim": 10, "eps": 1e-4})
        safe_action_batch = sol[:, :self.env.action_space.shape[0]]
        return safe_action_batch

    def cbf_layer(self, Qs, ps, Gs, hs, As=None, bs=None, solver_args=None):
        """

        Parameters
        ----------
        Qs : torch.Tensor
        ps : torch.Tensor
        Gs : torch.Tensor
            shape (batch_size, num_ineq_constraints, num_vars)
        hs : torch.Tensor
            shape (batch_size, num_ineq_constraints)
        As : torch.Tensor, optional
        bs : torch.Tensor, optional
        solver_args : dict, optional

        Returns
        -------
        result : torch.Tensor
            Result of QP
        """

        if solver_args is None:
            solver_args = {}

        if As is None or bs is None:
            As = torch.Tensor().to(self.device).double()
            bs = torch.Tensor().to(self.device).double()

        result = QPFunction(verbose=0, **solver_args)(Qs.double(), ps.double(), Gs.double(), hs.double(), As, bs).float()
        if torch.any(torch.isnan(result)):
            prRed('QP Failed to solve - result is nan == {}!'.format(torch.any(torch.isnan(result))))
            raise Exception('QP Failed to solve')
        return result

    def get_cbf_qp_constraints(self, state_batch, action_batch, mean_pred_batch, sigma_pred_batch, modular=False):
        """Build up matrices required to solve qp
        Program specifically solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Each control barrier certificate is of the form:
            dh/dx^T (f_out + g_out u) >= -gamma^b h_out^3 where out here is an output of the state.

        In the case of SafetyGym_point dynamics:
        state = [x y θ v ω]
        state_d = [v*cos(θ) v*sin(θ) omega ω u^v u^ω]

        Quick Note on batch matrix multiplication for matrices A and B:
            - Batch size should be first dim
            - Everything needs to be 3-dimensional
            - E.g. if B is a vec, i.e. shape (batch_size, vec_length) --> .view(batch_size, vec_length, 1)

        Parameters
        ----------
        state_batch : torch.tensor
            current state (check dynamics.py for details on each dynamics' specifics)
        action_batch : torch.tensor
            Nominal control input.
        mean_pred_batch : torch.tensor
            mean disturbance prediction state, dimensions (n_s, n_u)
        sigma_pred_batch : torch.tensor
            standard deviation in additive disturbance after undergoing the output dynamics.
        gamma_b : float, optional
            CBF parameter for the class-Kappa function

        Returns
        -------
        P : torch.tensor
            Quadratic cost matrix in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        q : torch.tensor
            Linear cost vector in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        G : torch.tensor
            Inequality constraint matrix (G[u,eps] <= h) of size (num_constraints, n_u + 1)
        h : torch.tensor
            Inequality constraint vector (G[u,eps] <= h) of size (num_constraints,)
        """

        assert len(state_batch.shape) == 2 and len(action_batch.shape) == 2 and len(mean_pred_batch.shape) == 2 and len(
            sigma_pred_batch.shape) == 2, print(state_batch.shape, action_batch.shape, mean_pred_batch.shape,
                                                sigma_pred_batch.shape)

        batch_size = state_batch.shape[0]
        gamma_b = self.gamma_b

        # Expand dims
        state_batch = torch.unsqueeze(state_batch, -1)
        action_batch = torch.unsqueeze(action_batch, -1)
        mean_pred_batch = torch.unsqueeze(mean_pred_batch, -1)
        sigma_pred_batch = torch.unsqueeze(sigma_pred_batch, -1)

        if self.env.dynamics_mode == 'Unicycle':

            num_cbfs = len(self.env.hazards)
            l_p = self.l_p
            buffer = 0.1

            thetas = state_batch[:, 2, :].squeeze(-1)
            c_thetas = torch.cos(thetas)
            s_thetas = torch.sin(thetas)

            # p(x): lookahead output (batch_size, 2)
            ps = torch.zeros((batch_size, 2)).to(self.device)
            ps[:, 0] = state_batch[:, 0, :].squeeze(-1) + l_p * c_thetas
            ps[:, 1] = state_batch[:, 1, :].squeeze(-1) + l_p * s_thetas

            # p_dot(x) = f_p(x) + g_p(x)u + D_p where f_p(x) = 0,  g_p(x) = RL and D_p is the disturbance

            # f_p(x) = [0,...,0]^T
            f_ps = torch.zeros((batch_size, 2, 1)).to(self.device)

            # g_p(x) = RL where L = diag([1, l_p])
            Rs = torch.zeros((batch_size, 2, 2)).to(self.device)
            Rs[:, 0, 0] = c_thetas
            Rs[:, 0, 1] = -s_thetas
            Rs[:, 1, 0] = s_thetas
            Rs[:, 1, 1] = c_thetas
            Ls = torch.zeros((batch_size, 2, 2)).to(self.device)
            Ls[:, 0, 0] = 1
            Ls[:, 1, 1] = l_p
            g_ps = torch.bmm(Rs, Ls)  # (batch_size, 2, 2)

            # D_p(x) = g_p [0 D_θ]^T + [D_x1 D_x2]^T
            mu_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
            mu_theta_aug[:, 1, :] = mean_pred_batch[:, 2, :]
            mu_ps = torch.bmm(g_ps, mu_theta_aug) + mean_pred_batch[:, :2, :]
            sigma_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
            sigma_theta_aug[:, 1, :] = sigma_pred_batch[:, 2, :]
            sigma_ps = torch.bmm(torch.abs(g_ps), sigma_theta_aug) + sigma_pred_batch[:, :2, :]

            # Build RCBFs
            hs = 1e3 * torch.ones((batch_size, num_cbfs), device=self.device)  # the RCBF itself
            dhdps = torch.zeros((batch_size, num_cbfs, 2), device=self.device)
            hazards = self.env.hazards
            for i in range(len(hazards)):
                if hazards[i]['type'] == 'circle':  # 1/2 * (||ps - x_obs||^2 - r^2)
                    obs_loc = to_tensor(hazards[i]['location'], torch.FloatTensor, self.device)
                    hs[:, i] = 0.5 * (torch.sum((ps - obs_loc)**2, dim=1) - (hazards[i]['radius'] + buffer)**2)
                    dhdps[:, i, :] = (ps - obs_loc)
                elif hazards[i]['type'] == 'polygon':  # max_j(h_j) where h_j = 1/2 * (dist2seg_j)^2
                    vertices = sort_vertices_cclockwise(hazards[i]['vertices'])  # (n_v, 2)
                    segments = np.diff(vertices, axis=0,
                                       append=vertices[[0]])  # (n_v, 2) at row i contains vector from v_i to v_i+1
                    segments = to_tensor(segments, torch.FloatTensor, self.device)
                    vertices = to_tensor(vertices, torch.FloatTensor, self.device)
                    # Get max RBCF TODO: Can be optimized
                    for j in range(segments.shape[0]):
                        # Compute Distances to segment
                        dot_products = torch.matmul(ps - vertices[j:j + 1], segments[j]) / torch.sum(
                            segments[j] ** 2)  # (batch_size,)
                        mask0_ = dot_products < 0  # if <0 closest point on segment is vertex j
                        mask1_ = dot_products > 1  # if >0 closest point on segment is vertex j+1
                        mask_ = torch.logical_and(dot_products >= 0,
                                                  dot_products <= 1)  # Else find distance to line l_{v_j, v_j+1}
                        # Compute Distances
                        dists2seg = torch.zeros((batch_size))
                        if mask0_.sum() > 0:
                            dists2seg[mask0_] = torch.linalg.norm(ps[mask0_] - vertices[[j]], dim=1)
                        if mask1_.sum() > 0:
                            dists2seg[mask1_] = torch.linalg.norm(ps[mask1_] - vertices[[(j + 1) % segments.shape[0]]], dim=1)
                        if mask_.sum() > 0:
                            dists2seg[mask_] = torch.linalg.norm(
                                dot_products[mask_, None] * segments[j].tile((torch.sum(mask_), 1)) + vertices[[j]] -
                            ps[mask_], dim=1)
                        # Compute hs_ for this segment
                        hs_ = 0.5 * ((dists2seg ** 2) + 0.5*buffer)  # (batch_size,)
                        # Compute dhdps TODO: Can be optimized to only compute for indices that need updating
                        dhdps_ = torch.zeros((batch_size, 2))
                        if mask0_.sum() > 0:
                            dhdps_[mask0_] = ps[mask0_] - vertices[[j]]
                        if mask1_.sum() > 0:
                            dhdps_[mask1_] = ps[mask1_] - vertices[[(j + 1) % segments.shape[0]]]
                        if mask_.sum() > 0:
                            normal_vec = torch.tensor([segments[j][1], -segments[j][0]])
                            normal_vec /= torch.linalg.norm(normal_vec)
                            dhdps_[mask_] = (ps[mask_]-vertices[j]).matmul(normal_vec) * normal_vec.view((1,2)).repeat(torch.sum(mask_), 1)  # dot products (batch_size, 1)
                        # Find indices to update (closest segment basically, worst case -> CBF boolean and is a min)
                        idxs_to_update = torch.nonzero(hs[:, i] - hs_ > 0)
                        # Update the actual hs to be used in the constraints
                        if idxs_to_update.shape[0] > 0:
                            hs[idxs_to_update, i] = hs_[idxs_to_update]
                            # Compute dhdhps for those indices
                            dhdps[idxs_to_update, i, :] = dhdps_[idxs_to_update, :]
                else:
                    raise Exception('Only obstacles of type `circle` or `polygon` are supported, got: {}'.format(hazards[i]['type']))

            n_u = action_batch.shape[1]  # dimension of control inputs
            num_constraints = num_cbfs + 2 * n_u  # each cbf is a constraint, and we need to add actuator constraints (n_u of them)

            # Inequality constraints (G[u, eps] <= h)
            G = torch.zeros((batch_size, num_constraints, n_u + 1)).to(self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
            h = torch.zeros((batch_size, num_constraints)).to(self.device)
            ineq_constraint_counter = 0

            # Add inequality constraints
            G[:, :num_cbfs, :n_u] = -torch.bmm(dhdps, g_ps)  # h1^Tg(x)
            G[:, :num_cbfs, n_u] = -1  # for slack
            h[:, :num_cbfs] = gamma_b * (hs ** 3) + (torch.bmm(dhdps, f_ps + mu_ps) - torch.bmm(torch.abs(dhdps), sigma_ps) + torch.bmm(torch.bmm(dhdps, g_ps), action_batch)).squeeze(-1)
            ineq_constraint_counter += num_cbfs

            # Let's also build the cost matrices, vectors to minimize control effort and penalize slack
            P = torch.diag(torch.tensor([1.e0, 1.e-2, 1e5])).repeat(batch_size, 1, 1).to(self.device)
            q = torch.zeros((batch_size, n_u + 1)).to(self.device)

        elif self.env.dynamics_mode == 'Pvtol':

            # # Get nearby obstacles (if batch_size is 1) TODO: Generalize this to any batch_size
            # if batch_size == 1:
            #     nearby_hazards = np.argwhere(np.sum((to_numpy(state_batch.squeeze()[:2]) - self.env.hazard_locations) ** 2, axis=1) < (6*self.env.hazards_radius) ** 2).squeeze(-1)
            #     hazards_locations = self.env.hazard_locations[nearby_hazards]
            #     hazards_radius = self.env.hazards_radius[nearby_hazards]
            # else:
            #     hazards_locations = self.env.hazard_locations
            #     hazards_radius = self.env.hazards_radius

            hazards_locations = self.env.hazard_locations
            hazards_radius = self.env.hazards_radius

            num_cbfs = 4 + 1 + 1
            if not modular:  # 4 for the arena, 1 for thrust and 1 for angle limits, and 1 for each obstacle
                num_cbfs += hazards_locations.shape[0]
            buffer = 0.3

            # Orientation
            thetas = state_batch[:, 2, :].squeeze(-1)
            c_thetas = torch.cos(thetas)
            s_thetas = torch.sin(thetas)

            # position
            ps = torch.zeros((batch_size, 2)).to(self.device)
            ps[:, 0] = state_batch[:, 0, :].squeeze(-1)
            ps[:, 1] = state_batch[:, 1, :].squeeze(-1)

            # velocities
            vs = torch.zeros((batch_size, 2)).to(self.device)
            vs[:, 0] = state_batch[:, 3, :].squeeze(-1)
            vs[:, 1] = state_batch[:, 4, :].squeeze(-1)

            # Thrust
            thrusts = state_batch[:, 5, :].squeeze(-1)

            n_u = action_batch.shape[1]  # dimension of control inputs
            num_constraints = num_cbfs + 2 * n_u  # each cbf is a constraint, and we need to add actuator constraints (n_u of them)

            # Inequality constraints (G[u, eps] <= h)
            G = torch.zeros((batch_size, num_constraints, n_u+4)).to(self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
            h = torch.zeros((batch_size, num_constraints)).to(self.device)
            ineq_constraint_counter = 0

            # Add Left boundary CBF
            gamma, gamma_2, gamma_3 = 1.5, 1.5, 1.5
            G[:, 0, 0] = s_thetas  # thrust_derivative
            G[:, 0, 1] = c_thetas * thrusts  # omega
            G[:, 0, n_u] = -1  # for slack
            h[:, 0] = (gamma+gamma_2+gamma_3)*(-s_thetas * thrusts)
            h[:, 0] += (gamma_3*(gamma_2 + gamma) + gamma_2 * gamma)*(vs[:, 0])
            h[:, 0] += gamma * gamma_2 * gamma_3 * (ps[:, 0] - self.env.bds[0, 0] - buffer) - (s_thetas * action_batch[:, 0, 0] + c_thetas * thrusts * action_batch[:, 1, 0])
            ineq_constraint_counter += 1

            # Add Right boundary CBF
            gamma, gamma_2, gamma_3 = 1.5, 1.5, 1.5
            G[:, 1, 0] = -s_thetas  # thrust_derivative
            G[:, 1, 1] = -c_thetas * thrusts  # omega
            G[:, 1, n_u] = -1  # for slack
            h[:, 1] = (gamma+gamma_2+gamma_3)*(s_thetas * thrusts)
            h[:, 1] += (gamma_3*(gamma_2 + gamma) + gamma_2 * gamma)*(-vs[:, 0])
            h[:, 1] += gamma * gamma_2 * gamma_3 * (self.env.bds[1, 0] - ps[:, 0] - buffer) + (s_thetas * action_batch[:, 0, 0] + c_thetas * thrusts * action_batch[:, 1, 0])
            ineq_constraint_counter += 1

            # Add Bottom boundary CBF
            gamma, gamma_2, gamma_3 = 1.5, 1.5, 1.5
            G[:, 2, 0] = -c_thetas  # thrust_derivative
            G[:, 2, 1] = s_thetas * thrusts  # omega
            G[:, 2, n_u] = -1  # for slack
            h[:, 2] = (gamma+gamma_2+gamma_3)*(c_thetas * thrusts - 1)  # hdd
            h[:, 2] += (gamma_3*(gamma_2 + gamma) + gamma_2 * gamma)*(vs[:, 1])  # hd
            h[:, 2] += gamma * gamma_2 * gamma_3 * (ps[:, 1] - self.env.bds[0, 1] - buffer) + (c_thetas * action_batch[:, 0, 0] - s_thetas * thrusts * action_batch[:, 1, 0])  # h
            ineq_constraint_counter += 1

            # Add Top boundary CBF
            gamma, gamma_2, gamma_3 = 1.5, 1.5, 1.5
            G[:, 3, 0] = c_thetas  # thrust_derivative
            G[:, 3, 1] = -s_thetas * thrusts  # omega
            G[:, 3, n_u] = -1  # for slack
            h[:, 3] = (gamma+gamma_2+gamma_3)*(-c_thetas * thrusts + 1)  # hdd
            h[:, 3] += (gamma_3*(gamma_2 + gamma) + gamma_2 * gamma)*(-vs[:, 1])  # hd
            h[:, 3] += gamma * gamma_2 * gamma_3 * (self.env.bds[1, 1] - ps[:, 1] - buffer) + (-c_thetas * action_batch[:, 0, 0] + s_thetas * thrusts * action_batch[:, 1, 0]) # h
            ineq_constraint_counter += 1

            # Add 45-degree constraint on theta (h = 0.5 * [(pi/3)**2 - theta^2])
            gamma = 2
            G[:, 4, 0] = 0  # thrust_derivative
            G[:, 4, 1] = thetas  # omega
            G[:, 4, n_u+1] = -1  # for slack
            h[:, 4] = gamma * 0.5 * ((np.pi/3.0)**2 - thetas**2) - thetas * action_batch[:, 1, 0]  # h
            ineq_constraint_counter += 1

            # Add thrust-limit (h = 0.5 * (thrust_limit**2 - (thrust-1)**2)
            gamma = 1
            G[:, 5, 0] = thrusts-1  # thrust_derivative
            G[:, 5, 1] = 0  # omega
            G[:, 5, n_u+2] = -1  # for slack
            h[:, 5] = gamma * (0.5 * (0.50**2 - (thrusts-1)**2)) - (thrusts - 1) * action_batch[:, 0, 0]  # h
            ineq_constraint_counter += 1

            # Obstacles
            if not modular:
                gamma, gamma_2, gamma_3 = 1.5, 1.5, 1.5
                for i in range(hazards_locations.shape[0]):
                    obs_loc = to_tensor(hazards_locations[i], torch.FloatTensor, self.device)
                    rel_vecs = (ps - obs_loc)
                    G[:, ineq_constraint_counter, 0] = -(rel_vecs[:, 0] * -s_thetas + rel_vecs[:, 1] * c_thetas)
                    G[:, ineq_constraint_counter, 1] = -thrusts * (rel_vecs[:, 0] * -c_thetas + rel_vecs[:, 1] * -s_thetas)
                    G[:, ineq_constraint_counter, n_u + 3] = -1
                    h[:, ineq_constraint_counter] = 3*(vs[:, 0] * -s_thetas * thrusts + vs[:, 1] * (c_thetas * thrusts - 1))  # hddd
                    h[:, ineq_constraint_counter] += (gamma * gamma_2 * gamma_3) * (torch.sum(vs**2, dim=1) + rel_vecs[:, 0] * -s_thetas*thrusts + rel_vecs[:, 1]*(c_thetas * thrusts - 1))
                    h[:, ineq_constraint_counter] += (gamma_3 * (gamma_2 + gamma) + gamma_2 * gamma) * (rel_vecs[:, 0] * vs[:, 0] + rel_vecs[:, 1] * vs[:, 1])
                    h[:, ineq_constraint_counter] += 0.5 * gamma_3 * gamma_2 * gamma * (torch.sum(rel_vecs**2, dim=1) - (1.05*hazards_radius[i])**2 - (1.3*buffer)**2)
                    h[:, ineq_constraint_counter] += (rel_vecs[:, 0] * -s_thetas + rel_vecs[:, 1] * c_thetas) * action_batch[:, 0, 0]
                    h[:, ineq_constraint_counter] += (thrusts * (rel_vecs[:, 0] * -c_thetas + rel_vecs[:, 1] * -s_thetas)) * action_batch[:, 1, 0]
                    ineq_constraint_counter += 1
            # Let's also build the cost matrices, vectors to minimize control effort and penalize slack
            P = torch.diag(1e5 * torch.ones(n_u + 4)).repeat(batch_size, 1, 1).to(self.device)
            P[:, 0, 0] = 1.5#0.3
            P[:, 1, 1] = 0.3#0.5
            q = torch.zeros((batch_size, n_u + 4)).to(self.device)
        else:
            raise Exception('Dynamics mode unknown!')

        # Add Actuator Constraints
        n_u = action_batch.shape[1]  # dimension of control inputs

        for c in range(n_u):

            # u_max >= u_nom + u ---> u <= u_max - u_nom
            if self.u_max is not None:
                G[:, ineq_constraint_counter, c] = 1
                h[:, ineq_constraint_counter] = self.u_max[c] - action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1

            # u_min <= u_nom + u ---> -u <= u_min - u_nom
            if self.u_min is not None:
                G[:, ineq_constraint_counter, c] = -1
                h[:, ineq_constraint_counter] = -self.u_min[c] + action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1

        return P, q, G, h

    def get_control_bounds(self):
        """

        Returns
        -------
        u_min : torch.tensor
            min control input.
        u_max : torch.tensor
            max control input.
        """

        u_min = torch.tensor(self.env.safe_action_space.low).to(self.device)
        u_max = torch.tensor(self.env.safe_action_space.high).to(self.device)

        return u_min, u_max


if __name__ == "__main__":

    from build_env import build_env
    from rcbf_sac.dynamics import DynamicsModel
    from copy import deepcopy
    from rcbf_sac.utils import to_numpy, prGreen


    def simple_controller(env, state, goal):
        goal_xy = goal[:2]
        goal_dist = -np.log(goal[2])  # the observation is np.exp(-goal_dist)
        v = 0.02 * goal_dist
        relative_theta = 1.0 * np.arctan2(goal_xy[1], goal_xy[0])
        omega = 1.0 * relative_theta

        return np.clip(np.array([v, omega]), env.action_space.low, env.action_space.high)


    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # Environment Args
    parser.add_argument('--env-name', default="SafetyGym", help='Options are Unicycle or SafetyGym')
    parser.add_argument('--robot_xml', default='xmls/point.xml',
                        help="SafetyGym Currently only supporting xmls/point.xml")
    parser.add_argument('--k_d', default=3.0, type=float)
    parser.add_argument('--gamma_b', default=100, type=float)
    parser.add_argument('--l_p', default=0.03, type=float)
    parser.add_argument('--gp_model_size', default=2000, type=int, help='gp')
    parser.add_argument('--cuda', action='store_true', help='run on CUDA (default: False)')
    args = parser.parse_args()
    # Environment
    env = build_env(args)

    device = torch.device('cuda' if args.cuda else 'cpu')


    def to_def_tensor(ndarray):

        return to_tensor(ndarray, torch.FloatTensor, device)


    diff_cbf_layer = CBFQPLayer(env, args, args.gamma_b, args.k_d, args.l_p)
    dynamics_model = DynamicsModel(env, args)

    obs = env.reset()
    done = False

    ep_ret = 0
    ep_cost = 0
    ep_step = 0

    for i_step in range(3000):

        if done:
            prGreen('Episode Return: %.3f \t Episode Cost: %.3f' % (ep_ret, ep_cost))
            ep_ret, ep_cost, ep_step = 0, 0, 0
            obs = env.reset()

        state = dynamics_model.get_state(obs)
        disturb_mean, disturb_std = dynamics_model.predict_disturbance(state)

        action = simple_controller(env, state, obs[-3:])  # TODO: observations last 3 indicated
        # action = 2*np.random.rand(2) - 1.0
        assert env.action_space.contains(action)
        final_action = diff_cbf_layer.get_safe_action(to_def_tensor(state), to_def_tensor(action),
                                                      to_def_tensor(disturb_mean), to_def_tensor(disturb_std))
        final_action = to_numpy(final_action)

        # Env Step
        observation2, reward, done, info = env.step(final_action)
        observation2 = deepcopy(observation2)

        # Update state and store transition for GP model learning
        next_state = dynamics_model.get_state(observation2)
        if ep_step % 2 == 0:
            dynamics_model.append_transition(state, final_action, next_state)

        # print('reward', reward)
        ep_ret += reward
        ep_cost += info.get('cost', 0)
        ep_step += 1
        # env.render()

        obs = observation2
        state = next_state