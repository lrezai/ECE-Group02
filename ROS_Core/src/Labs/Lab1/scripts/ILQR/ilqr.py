from typing import Tuple, Optional, Dict, Union
from jaxlib.xla_extension import DeviceArray
import time
import os
import numpy as np
import jax
from .dynamics import Bicycle5D
from .cost import Cost, CollisionChecker, Obstacle
from .ref_path import RefPath
from .config import Config
import time

status_lookup = ['Iteration Limit Exceed',
				 'Converged',
				 'Failed Line Search']


class ILQR():
	def __init__(self, config_file=None) -> None:

		self.config = Config()  # Load default config.
		if config_file is not None:
			self.config.load_config(config_file)  # Load config from file.

		self.load_parameters()
		print('ILQR setting:', self.config)

		# Set up Jax parameters
		jax.config.update('jax_platform_name', self.config.platform)
		print('Jax using Platform: ', jax.lib.xla_bridge.get_backend().platform)

		# If you want to use GPU, lower the memory fraction from 90% to avoid OOM.
		os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '20'

		self.dyn = Bicycle5D(self.config)
		self.cost = Cost(self.config)
		self.ref_path = None

		# collision checker
		# Note: This will not be used until lab2.
		self.collision_checker = CollisionChecker(self.config)
		self.obstacle_list = []

		# Do a dummy run to warm up the jitted functions.
		self.warm_up()

	def load_parameters(self):
		'''
		This function defines ILQR parameters from <self.config>.
		'''
		# ILQR parameters
		self.dim_x = self.config.num_dim_x
		self.dim_u = self.config.num_dim_u
		self.T = int(self.config.T)
		self.dt = float(self.config.dt)
		self.max_iter = int(self.config.max_iter)
		self.tol = float(self.config.tol)  # ILQR update tolerance.

		# line search parameters.
		self.alphas = self.config.line_search_base**(
			np.arange(self.config.line_search_a,
												  self.config.line_search_b,
												  self.config.line_search_c)
		)

		print('Line Search Alphas: ', self.alphas)

		# regularization parameters
		self.reg_min = float(self.config.reg_min)
		self.reg_max = float(self.config.reg_max)
		self.reg_init = float(self.config.reg_init)
		self.reg_scale_up = float(self.config.reg_scale_up)
		self.reg_scale_down = float(self.config.reg_scale_down)
		self.max_attempt = self.config.max_attempt

	def warm_up(self):
		'''
		Warm up the jitted functions.
		'''
		# Build a fake path as a 1 meter radius circle.
		theta = np.linspace(0, 2 * np.pi, 100)
		centerline = np.zeros([2, 100])
		centerline[0, :] = 1 * np.cos(theta)
		centerline[1, :] = 1 * np.sin(theta)

		self.ref_path = RefPath(centerline, 0.5, 0.5, 1, True)

		# add obstacle
		obs = np.array([[0, 0, 0.5, 0.5], [1, 1.5, 1, 1.5]]).T
		obs_list = [[obs for _ in range(self.T)]]
		self.update_obstacles(obs_list)

		x_init = np.array([0.0, -1.0, 1, 0, 0])
		print('Start warm up ILQR...')
		self.plan(x_init)
		print('ILQR warm up finished.')

		self.ref_path = None
		self.obstacle_list = []

	def update_ref_path(self, ref_path: RefPath):
		'''
		Update the reference path.
		Args:
				ref_path: RefPath: reference path.
		'''
		self.ref_path = ref_path

	def update_obstacles(self, vertices_list: list):
		'''
		Update the obstacle list for a list of vertices.
		Args:
				vertices_list: list of np.ndarray: list of vertices for each obstacle.
		'''
		# Note: This will not be used until lab2.
		self.obstacle_list = []
		for vertices in vertices_list:
			self.obstacle_list.append(Obstacle(vertices))

	def get_references(self, trajectory: Union[np.ndarray, DeviceArray]):
		'''
		Given the trajectory, get the path reference and obstacle information.
		Args:
				trajectory: [num_dim_x, T] trajectory.
		Returns:
				path_refs: [num_dim_x, T] np.ndarray: references.
				obs_refs: [num_dim_x, T] np.ndarray: obstacle references.
		'''
		trajectory = np.asarray(trajectory)
		path_refs = self.ref_path.get_reference(trajectory[:2, :])
		obs_refs = self.collision_checker.check_collisions(
			trajectory, self.obstacle_list)
		return path_refs, obs_refs

	def plan(self, init_state: np.ndarray,
			 controls: Optional[np.ndarray] = None) -> Dict:
		'''
		Main ILQR loop.
		Args:
				init_state: [num_dim_x] np.ndarray: initial state.
				control: [num_dim_u, T] np.ndarray: initial control.
		Returns:
				A dictionary with the following keys:
						status: int: -1 for failure, 0 for success. You can add more status if you want.
						t_process: float: time spent on planning.
						trajectory: [num_dim_x, T] np.ndarray: ILQR planned trajectory.
						controls: [num_dim_u, T] np.ndarray: ILQR planned controls sequence.
						K_closed_loop: [num_dim_u, num_dim_x, T] np.ndarray: closed loop gain.
						k_closed_loop: [num_dim_u, T] np.ndarray: closed loop bias.
		'''

		# We first check if the planner is ready
		if self.ref_path is None:
			print('No reference path is provided.')
			return dict(status=-1)

		# if no initial control sequence is provided, we assume it is all zeros.
		if controls is None:
			controls = np.zeros((self.dim_u, self.T))
		else:
			assert controls.shape[1] == self.T

		# Start timing
		t_start = time.time()

		# Rolls out the nominal trajectory and gets the initial cost.
		trajectory, controls = self.dyn.rollout_nominal_np(
			init_state, controls)

		# Get path and obstacle references based on your current nominal trajectory.
		# Note: you will NEED TO call this function and get new references at each iteration.
		path_refs, obs_refs = self.get_references(trajectory)

		# Get the initial cost of the trajectory.
		J = self.cost.get_traj_cost(trajectory, controls, path_refs, obs_refs)

		##########################################################################
		# TODO 1: Implement the ILQR algorithm. Feel free to add any helper functions.
		# You will find following implemented functions useful:

		# ******** Functions to compute the Jacobians of the dynamics  ************
		# A, B = self.dyn.get_jacobian_np(trajectory, controls)

		# Returns the linearized 'A' and 'B' matrix of the ego vehicle around
		# nominal trajectory and controls.

		# Args:
		# 	trajectory: np.ndarray, (dim_x, T) trajectory along the nominal trajectory.
		# 	controls: np.ndarray, (dim_u, T) controls along the trajectory.

		# Returns:
		# 	A: np.ndarray, (dim_x, T) the Jacobian of the dynamics w.r.t. the state.
		# 	B: np.ndarray, (dim_u, T) the Jacobian of the dynamics w.r.t. the control.

		# ******** Functions to roll the dynamics for one step  ************
		# state_next, control_clip = self.dyn.integrate_forward_np(state, control)

		# Finds the next state of the vehicle given the current state and
		# control input.

		# Args:
		# 	state: np.ndarray, (dim_x).
		# 	control: np.ndarray, (dim_u).

		# Returns:
		# 	state_next: np.ndarray, (dim_x) next state.
		# 	control_clip: np.ndarray, (dim_u) clipped control.

		# *** Functions to get total cost of a trajectory and control sequence  ***
		# J = self.cost.get_traj_cost(trajectory, controls, path_refs, obs_refs)
		# Given the trajectory, control seq, and references, return the sum of the cost.
		# Input:
		# 	trajectory: (dim_x, T) array of state trajectory
		# 	controls:   (dim_u, T) array of control sequence
		# 	path_refs:  (dim_ref, T) array of references (e.g. reference path, reference velocity, etc.)
		# 	obs_refs: *Optional* (num_obstacle, (2, T)) List of obstacles. Default to None
		# return:
		# 	cost: float, sum of the running cost over the trajectory

		# ******** Functions to get jacobian and hessian of the cost ************
		# q, r, Q, R, H = self.cost.get_derivatives_np(trajectory, controls, path_refs, obs_refs)

		# Given the trajectory, control seq, and references, return Jacobians and Hessians of cost function
		# Input:
		# 	trajectory: (dim_x, T) array of state trajectory
		# 	controls:   (dim_u, T) array of control sequence
		# 	path_refs:  (dim_ref, T) array of references (e.g. reference path, reference velocity, etc.)
		# 	obs_refs: *Optional* (num_obstacle, (2, T)) List of obstacles. Default to None
		# return:
		# 	q: np.ndarray, (dim_x, T) jacobian of cost function w.r.t. states
	#   r: np.ndarray, (dim_u, T) jacobian of cost function w.r.t. controls
	#   Q: np.ndarray, (dim_x, dim_u, T) hessian of cost function w.r.t. states
	#   R: np.ndarray, (dim_u, dim_u, T) hessian of cost function w.r.t. controls
	#   H: np.ndarray, (dim_x, dim_u, T) hessian of cost function w.r.t. states and controls

		########################### #END of TODO 1 #####################################
		reg = self.reg_init
		steps = self.max_attempt
		X = trajectory
		U = controls
		converged = False
		for i in range(steps):
			K_closed_loop, k_open_loop, reg = self.backward_pass(X, U, reg, path_refs, obs_refs)
			changed = False
   
			for alpha in self.alphas:
				X_new, U_new = self.roll_out(0,0,0,0,0,0)
       			#X_new, U_new = self.roll_out(X, J, U, K_closed, k_openloop, alpha)
				path_refs, obs_refs = self.get_references(X_new)
				J_new = self.cost.get_traj_cost(X_new, U_new, path_refs, obs_refs)
    
		if J_new <= J:
			if np.abs(J - J_new) < self.tol:
				converged = True
				J = J_new
				X = X_new
				U = U_new
				changed = True
				#break
		if not changed:
			print("line search failed with reg = ", reg, " at step ", i)
			#break
		if converged:
			print("converged after ", i, " steps.")
			#break

		trajectory = X
		controls = U

		t_process = time.time() - t_start
		solver_info = dict(
			t_process=t_process,  # Time spent on planning
			trajectory=trajectory,
			controls=controls,
						status=changed,  # TODO: Fill this in
			K_closed_loop=K_closed_loop,  # TODO: Fill this in
			k_open_loop=k_open_loop  # TODO: Fill this in
			# Optional TODO: Fill in other information you want to return
		)
		return solver_info

	def roll_out(self, init_state, cost, controls, big_Kt, little_kt, alpha):
		# epsilon = 0.5
		# for alpha in alphas:
		X = np.zeros_like(init_state)
		U = np.zeros_like(controls)

		X[:, 0] = init_state[:, 0]
		T = init_state.shape[1]
		for t in range(T-1):
			K = big_Kt[:, :, t]
			k = little_kt[:, t]
			U[:, t] = controls[:, t] + alpha*k + \
				K @ (X[:, t] - init_state[:, t])
			X[:, t+1], U[:,
				t] = self.dyn.integrate_forward_np(X[:, t], U[:, t])
		return X, U
		   # path_refs, obs_refs = self.get_references(X)
		   # J = self.cost.get_traj_cost(X, U, path_refs, obs_refs)
		   # if J < cost:
		   # 	break
		# return X, U, J, path_refs, obs_refs

	def backward_pass(self, X, U, reg, path_refs, obs_refs):
		q, r, Q, R, H = self.get_derivatives_np(X, U, path_refs, obs_refs)
		A, B = self.dyn.get_jacobian_np(X, U)
		T = X.shape[1]
		k_open_loop = np.zeros((2, T))
		K_closed_loop = np.zeros((2, 5, T))

		# 2. Initialize pT = qT and PT = QT
		p = q[:, T-1]
		P = Q[:, T-1]

		# 3. t = T-1
		t = T-2

		# 4. while t ≥ 0
		while t >= 0:

			# 6. Done below
			Q_x = q[:, t] + A[:, :, t].T @ p
			Q_u = r[:, t] + B[:, :, t].T @ p
			Q_xx = Q[:, :, t] + A[:, :, t].T @ P @ A[:, :, t]
			Q_uu = R[:, :, t] + B[:, :, t].T @ P @ B[:, :, t]
			Q_ux = H[:, :, t] + B[:, :, t].T @ P @ A[:, :, t]

		   # 7. Compute regulatrized Hessian
			reg_matrix = reg*np.eye(5)
			Q_uu_reg = R[:, :, t] + B[:, :, t].T @ (P+reg_matrix) @ B[:, :, t]
			Q_ux_reg = H[:, :, t] + B[:, :, t].T @ (P+reg_matrix) @ A[:, :, t]

		   # 8. If Q_uu NOT positive definite
		if not np.all(np.linalg.eigvals(Q_uu_reg) > 0) and reg < 1e5:
				reg *= self.reg_scale_up
				t = T-2
				#continue
			# 13. Compute optimal control gain
		Q_uu_reg_inv = np.linalg.inv(Q_uu_reg)
		# Calculate policy
		k = -Q_uu_reg_inv @ Q_u
		K = -Q_uu_reg_inv @ Q_ux_reg
		k_open_loop[:, t] = k          
		K_closed_loop[:, :, t] = K

		   # Computing derivative and Hessian of V_t
		p = Q_x + K.T @ Q_uu @ k + K.T@Q_u + Q_ux.T@k
		P = Q_xx + K.T @ Q_uu @ K + K.T@Q_ux + Q_ux.T@K
		t -= 1

		reg = max(self.reg_min, reg/self.reg_scale_down)
		return K_closed_loop, k_open_loop, reg
