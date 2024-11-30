import torch
from typing import Any

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.common import VecEnvObs, VecEnvStepReturn
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# Debug drawing
try:
    from omni.isaac.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
except ImportError:
    pass

from tasks.push_t.pusht_env import compute_quat_angle, gen_rot_around_z
from tasks.push_maze.pushmaze_env_cfg import PushMazeEnvCfg
from tasks.push_maze.gen_maze_states import MazeA, MazeB, MazeC, generate, generate_robot_obj_pos
from utils.misc import AverageScalarMeter, to_torch


class PushMazeEnv(DirectRLEnv):
    cfg: PushMazeEnvCfg
    def __init__(self, cfg: PushMazeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Create Maze class object (to sample collision-free states in the PRM algorithm) and position limits
        if self.cfg.maze == "maze_a":
            self.maze_object = MazeA()
            self.dof_pos_upper_lim = [1.2, 1.2]
            self.dof_pos_lower_lim = [-1.2, -1.2]
        elif self.cfg.maze == "maze_b":
            self.maze_object = MazeB()
            self.dof_pos_upper_lim = [0.9, 0.9]
            self.dof_pos_lower_lim = [-0.9, -0.9]
        elif self.cfg.maze == "maze_c":
            self.maze_object = MazeC()
            self.dof_pos_upper_lim = [1.2, 1.2]
            self.dof_pos_lower_lim = [-1.2, -1.2]
        else:
            raise ValueError(f"Invalid maze name: {self.cfg.maze}")

        # Setup velocity limits
        self.dof_vel_lower_lim = torch.tensor(self.cfg.dof_vel_lower_limit, dtype=torch.float, device=self.device)
        self.dof_vel_upper_lim = torch.tensor(self.cfg.dof_vel_upper_limit, dtype=torch.float, device=self.device)

        # setup joint limits
        self.target_limit_lower = torch.tensor(self.dof_pos_lower_lim, dtype=torch.float, device=self.device)
        self.target_limit_upper = torch.tensor(self.dof_pos_upper_lim, dtype=torch.float, device=self.device)

        # set controller params (PD controller)
        self.kp = self.cfg.kp
        self.kd = self.cfg.kd

        # create target joint position to set the actions with PD controller
        self.target_joint_pos = None

        # create goal position
        self.goal = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)

        # Flags to control reset behavior as the reset behavior needs
        # to alternate between PRM and RL
        self.enable_reset = True
        self.reset_dist_type = "eval"

        # set up reset distribution (only used in on-policy learning)
        self.reset_state_buf = None

        # planning state (x) dimension
        self.planning_state_dim = 15  # 2 for robot position, 2 for robot velocity, 2 for object position, 4 for object orientation and 6 for object linear and angular velocity
        self.planner_goal_dim = 2  # 2 for goal position

        # set up goal buffer from the reset distribution
        self.goal_buf = to_torch(generate(self.maze_object, nsample=5000, pad=0.05), device=self.device)[:, :2]

        # set up valid position buffer for object and robot
        # Note that the pad is larger than that in goal buffer since we don't want the object to be stuck in the corner
        # And the robot stays close enough to the object
        self.robot_obj_pos_buf = to_torch(
            generate_robot_obj_pos(self.maze_object, nsample=5000, pad=0.1, min_dist=0.05, max_dist=0.1),
            device=self.device
        )

        # Logging success rate
        self.success = torch.zeros_like(self.reset_buf, dtype=torch.float)
        self.success_rate = AverageScalarMeter(100)
        self.extras["success_rate"] = 0.0

    def _setup_scene(self):
        """ Setup the scene with the robot, ground plane, and lights. """
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """ Pre-process actions before stepping through the physics. """
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """ Apply the actions to the robot using a PD controller. """
        self.joint_pos_target = torch.clamp(self.joint_pos + self.actions, self.target_limit_lower, self.target_limit_upper)
        force = (self.joint_pos_target - self.joint_pos) * self.kp - self.joint_vel * self.kd
        self.robot.set_joint_effort_target(force)

    def _get_observations(self) -> dict:
        """ Compute and return the observations for the environment.

        Returns:
            The observations (key: "policy") and states ((key: "critic") for the environment.
        """
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        obs_policy = torch.cat(
            (
                self.robot.data.joint_pos.view(self.num_envs, -1),  # robot position (2)
                self.robot.data.joint_vel.view(self.num_envs, -1),  # robot velocity (2)
                self.object_pos_xy,  # object x and y coordinate (2)
                self.object_rotation,  # object rotation (4)
                self.object_lin_vel_xy,  # object x and y velocity (2)
                self.object_ang_vel,  # object angular velocity (3)
                self.goal  # goal x and y coordinate (2)
            ),
            dim=-1,
        )
        obs_critic = obs_policy.clone()
        return {"policy": obs_policy, "critic": obs_critic}

    def get_observations(self) -> dict:
        """ A public function to access the observations and states of the environment. """
        return self._get_observations()

    def _get_rewards(self) -> torch.Tensor:
        """ Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).
        """
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        # Compute reward and distance between object and goal positions
        total_reward, dist = compute_rewards(
            self.object_pos_xy,
            self.goal,
            self.cfg.reward_type,
            self.cfg.dense_reward_scale,
            self.cfg.success_reward_scale,
            self.cfg.at_goal_threshold,
        )
        # Update success rate
        if self.reset_dist_type == "eval":
            self.success = torch.logical_or(self.success > 0, (dist < self.cfg.at_goal_threshold))
            self.extras["success"] = self.success

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """ Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Shape of individual tensors is (num_envs,).
        """
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # No explicit termination condition
        done = torch.zeros_like(self.reset_buf, dtype=torch.bool)

        return done, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """ Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.reset_dist_type == "eval":
            self.success_rate.update(self.success[env_ids])
            self.extras["success_rate"] = self.success_rate.get_mean()
            self.success[env_ids] = 0.0

        # reset goal position
        self.goal[env_ids] = self.goal_buf[torch.randint_like(env_ids, len(self.goal_buf))]

        # clear the previous goal and draw new goal
        env_goals = self.goal.cpu().numpy() + self.scene.env_origins[:, :2].cpu().numpy()
        try:
            # draw.clear_lines()
            draw_plus(points=env_goals, color="r")
        except:
            pass

        # create the new state for the environment
        if self.reset_dist_type == "train":
            # sample random state from the reset distribution
            sampled_idx = torch.randint(0, self.reset_state_buf.shape[0], (len(env_ids),))
            states = self.reset_state_buf[sampled_idx].to(self.device)
        else:
            robot_state = torch.zeros((len(env_ids), 4), device=self.device)
            # sample random position for object
            object_pos_xy = torch.tensor([0, 0.1], device=self.device).repeat(len(env_ids), 1)
            obj_rot = torch.tensor([1, 0, 0, 0], device=self.device).repeat(len(env_ids), 1)
            object_lin_vel_xy = torch.zeros((len(env_ids), 2), device=self.device)
            obj_ang_vel = torch.zeros((len(env_ids), 3), device=self.device)

            # Set the states to the environment
            states = torch.cat(
                [
                    robot_state,
                    object_pos_xy,
                    obj_rot,
                    object_lin_vel_xy,
                    obj_ang_vel,
                ],
                dim=1
            )
        self.set_env_states(states, env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values()

    def reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """ A public function to access the observations and states of the environment. """
        self._reset_idx(env_ids)

    def step_without_reset(self, action: torch.Tensor) -> VecEnvStepReturn:
        """ Execute one time-step of the environment's dynamics without resetting the environment.
            Almost the same as the step() function, but remove the environment reset.
            It would be useful for the PRM planner to simulate the environment without resetting it.
        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action.clone())
        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _compute_intermediate_values(self) -> None:
        # fetch joint position and joint velocity from the robot
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # compute the object x and y coordinate
        self.object_pos_xy = (self.object.data.root_pos_w - self.scene.env_origins)[:, 0:2]
        self.object_height = (self.object.data.root_pos_w - self.scene.env_origins)[:, 2]
        self.object_rotation = self.object.data.root_quat_w
        # compute the object linear and angular velocity
        self.object_lin_vel_xy = self.object.data.root_lin_vel_w[:, 0:2]
        self.object_ang_vel = self.object.data.root_ang_vel_w

    def compute_intermediate_values(self) -> None:
        """ A public function to update intermediate values of the environment. """
        self._compute_intermediate_values()

    def simulate(self) -> None:
        """ Simulate the environment for one step. """
        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            # set new state and goal into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

    def set_reset_dist_type(self, reset_dist_type: str) -> None:
        """ Set the reset distribution type. """
        self.reset_dist_type = reset_dist_type

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros(
            [self.num_envs, self.cfg.num_actions],
            dtype=torch.float32,
            device=self.device,
        )

        return actions

    def random_actions(self) -> torch.Tensor:
        """ Returns a buffer with random actions drawn from normal distribution

        Returns:
            torch.Tensor: A buffer of random actions torch actions
        """
        mean = torch.zeros(
            [self.num_envs, self.cfg.num_actions],
            dtype=torch.float32,
            device=self.device,
        )
        std = torch.ones(
            [self.num_envs, self.cfg.num_actions],
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.normal(mean, std)

        return actions

    def set_reset_state_buf(self, buf: torch.Tensor) -> None:
        self.reset_state_buf = buf

    def save_episode_context(self):
        """ Saves episode context to switch to planner """
        context = {
            "episode_length_buf": self.episode_length_buf.detach().clone(),
            "reset_buf": self.reset_buf.detach().clone(),
            "reset_terminated": self.reset_terminated.detach().clone(),
            "reset_time_outs": self.reset_time_outs.detach().clone(),
            "env_states": self.get_env_states(),
            "goal": self.goal.detach().clone() if hasattr(self, "goal") else "None",
        }

        return context

    def restore_episode_context(self, context):
        """ Restore episode context from planning to learning """
        with torch.no_grad():
            self.episode_length_buf = context["episode_length_buf"]
            self.reset_buf = context["reset_buf"]
            self.reset_terminated = context["reset_terminated"]
            self.reset_time_outs = context["reset_time_outs"]
            with torch.inference_mode():
                self.set_env_states(context["env_states"], torch.arange(self.num_envs, device=self.device))
            if hasattr(self, "goal"):
                self.goal = context["goal"]
            self.get_observations()

        return self.get_observations()


    """ PRM Planner functions """


    def get_env_q(self) -> torch.Tensor:
        """ Returns the current q_state of the environment """
        return self.get_env_states()

    def get_env_states(self) -> torch.Tensor:
        """ Returns the current state of the environment """
        self._compute_intermediate_values()

        return torch.cat(
            [
                self.joint_pos.detach().clone(),  # (2)
                self.joint_vel.detach().clone(),  # (2)
                self.object_pos_xy.detach().clone(),  # (2)
                self.object_rotation.detach().clone(),  # (4)
                self.object_lin_vel_xy.detach().clone(),  # (2)
                self.object_ang_vel.detach().clone(),  # (3)
            ],
            dim=1
        )

    def set_env_states(self, states, env_ids: torch.Tensor) -> None:
        """ Sets the state of the envs specified by env_idx """
        # Extract joint position and velocity from the states
        joint_pos = states[:, :2]
        joint_vel = states[:, 2:4]
        object_pos_xy = states[:, 4:6]
        object_rot = states[:, 6:10]
        object_lin_vel_xy = states[:, 10:12]
        object_ang_vel = states[:, 12:15]

        # Write the joint state to the simulation
        self.joint_pos_target = joint_pos
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Write object position and velocity to the simulation
        object_state = self.object.data.default_root_state.clone()[env_ids]
        object_state[:, :2] = object_pos_xy
        object_state[:, 3:7] = object_rot
        object_state[:, 7:9] = object_lin_vel_xy
        object_state[:, 10:13] = object_ang_vel
        # Add the origin of each environment
        object_state[:, :3] += self.scene.env_origins[env_ids]

        self.object.write_root_state_to_sim(object_state, env_ids)

    def set_goal(self, goals: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.goal[env_ids, :] = goals

    def q_to_goal(self, q: torch.Tensor) -> torch.Tensor:
        """ Extract goal position from q_state """
        goal = q[:, 4:6]  # extract object x and y position
        return goal

    def compute_goal_distance(self, prm_nodes: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
            Computes distance in goal state from a specific node to each node in node set.
        """
        # In the PushMaze task, the goal is the position of the object
        nodes_pos = prm_nodes[:, 4:6].view(-1, 2)  # extract position from the node
        distances = torch.linalg.norm(nodes_pos - goal[:2], dim=1)
        return distances

    def compute_distance(self, selected_node: torch.Tensor, prm_nodes: torch.Tensor) -> torch.Tensor:
        """ Computes distance from a specific node to each node in node set """
        # Robot Position distance
        robot_pos_dist = 1.0 * torch.linalg.norm(prm_nodes[:, 0:2] - selected_node[0:2], dim=1)
        # Robot Velocity distance
        robot_vel_dist = 0.05 * torch.linalg.norm(prm_nodes[:, 2:4] - selected_node[2:4], dim=1)
        # Object Position distance
        object_pos_dist = 1.0 * torch.linalg.norm(prm_nodes[:, 4:6] - selected_node[4:6], dim=1)
        # Object rotation distance
        object_rot_dist = 0.2 * compute_quat_angle(selected_node[6:10], prm_nodes[:, 6:10]).squeeze()
        # Object linear velocity distance
        object_lin_vel_xy_dist = 0.01 * torch.linalg.norm(prm_nodes[:, 10:12] - selected_node[10:12], dim=1)
        # Object angular velocity distance
        object_ang_vel_dist = 0.01 * torch.linalg.norm(prm_nodes[:, 12:15] - selected_node[12:15], dim=1)
        # Compute the total distance
        total_dist = robot_pos_dist + robot_vel_dist + object_pos_dist + object_rot_dist + object_lin_vel_xy_dist + object_ang_vel_dist
        return total_dist

    def sample_q(self, num_samples: int = 32) -> torch.Tensor:
        """ Uniformly sample initial collision-free nodes to be added to the graph """
        # Sample valid object positions in maze
        rand_idx = torch.randint(0, self.robot_obj_pos_buf.shape[0], (num_samples,))
        obj_pos_xy = self.robot_obj_pos_buf[rand_idx, :2]
        robot_pos = self.robot_obj_pos_buf[rand_idx, 2:4]
        # Sample random robot velocity within velocity limits
        alpha = torch_rand_float(
            0.0,
            1.0,
            (num_samples, self.dof_vel_lower_lim.shape[0]),
            device=self.device,
        )
        robot_vel = alpha * self.dof_vel_upper_lim + (1 - alpha) * self.dof_vel_lower_lim
        # Sample random object orientation
        obj_rot = gen_rot_around_z(num_samples, device=self.device)
        # Sample random object linear velocity within velocity limits
        obj_lin_vel_xy = torch.zeros((num_samples, 2), device=self.device)
        obj_ang_vel = torch.zeros((num_samples, 3), device=self.device)
        x_start = torch.cat(
            [
                robot_pos,
                robot_vel,
                obj_pos_xy,
                obj_rot,
                obj_lin_vel_xy,
                obj_ang_vel,
            ],
            dim=1
        )
        return x_start

    def sample_random_nodes(self, N: int = 32) -> torch.Tensor:
        """ Uniformly sample initial collision-free nodes to be added to the graph """
        sampled_nodes = []

        while len(sampled_nodes) < N:
            # sample random states unifromly
            x_samples = self.sample_q(num_samples=self.num_envs)
            # apply the sampled states to the environment
            with torch.inference_mode():
                self.set_env_states(x_samples, torch.tensor(list(range(self.num_envs)), device=self.device))
                self.simulate()
            self._compute_intermediate_values()

            # perform validity check
            invalid, x_start_prime = self.is_invalid()

            # add valid states to the list
            valid_indices = torch.nonzero(torch.logical_not(invalid), as_tuple=False).squeeze(-1)
            for idx in valid_indices:
                if len(sampled_nodes) >= N:
                    break
                sampled_nodes.append(x_start_prime[idx].clone())

        return torch.stack(sampled_nodes).to(self.device)

    def sample_random_goal_state(self, num_goal) -> torch.Tensor:
        """ Sample goal positions which is close to the nodes in the node set """
        return self.sample_random_nodes(num_goal)

    def is_invalid(self, debug: bool = False) -> (torch.Tensor, torch.Tensor):
        """ Check if the sampled state is valid to be added to the graph """
        # No position constraints in the PushMaze task as the collison is handled by the physics engine
        # velocity constraints
        invalid = torch.logical_or(
            torch.any(self.joint_vel < self.dof_vel_lower_lim, dim=1),
            torch.any(self.joint_vel > self.dof_vel_upper_lim, dim=1),
        )
        # object height constraints
        invalid = torch.where(
            self.object_height > self.cfg.height_limit,
            torch.ones_like(self.reset_buf),
            invalid,
        )
        # robot-object distance constraints
        obj_robot_dist = torch.linalg.norm(self.object_pos_xy - self.joint_pos, dim=1)
        invalid = torch.where(
            obj_robot_dist > self.cfg.valid_robot_object_thres,
            torch.ones_like(self.reset_buf),
            invalid,
        )

        x_start_prime = self.get_env_states()
        return invalid, x_start_prime


##
# torch.jit functions
##


@torch.jit.script
def compute_rewards(
    object_pos_xy: torch.Tensor,
    goal: torch.Tensor,
    reward_type: str,
    dense_reward_scale: float,
    success_reward_scale: float,
    at_goal_threshold: float,
):
    # distance between object position and goal position
    dist = torch.linalg.norm(object_pos_xy - goal, dim=1)
    # dense reward
    dense_reward = -(dist**2) * dense_reward_scale
    # success (sparse) reward
    success_reward = (dist < at_goal_threshold) * success_reward_scale

    if reward_type == "dense":
        reward = dense_reward
    elif reward_type == "sparse":
        reward = success_reward
    elif reward_type == "mixed":
        reward = dense_reward + success_reward
    else:
        raise ValueError(f"Invalid reward type: {reward_type}")

    return reward, dist

@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


##
# Debug drawing
##


def draw_plus(points, color="r"):
    """ Draw the goal position as a red cross in the maze """
    cmap = {"r": [1, 0, 0, 1], "g": [0, 1, 0, 1], "b": [0, 0, 1, 1]}
    color_rgba = cmap[color]
    for point in points:
        point_list_0 = [(point[0] - 0.03, point[1], 0.01), (point[0], point[1] - 0.03, 0.01)]
        point_list_1 = [(point[0] + 0.03, point[1], 0.01), (point[0], point[1] + 0.03, 0.01)]
        colors = [color_rgba for _ in range(2)]
        widths = [1.0 for _ in range(2)]
        try:
            draw.draw_lines(point_list_0, point_list_1, colors, widths)
        except:
            pass



