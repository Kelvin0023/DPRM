import math
import torch
from typing import Any

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.common import VecEnvObs, VecEnvStepReturn
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils.math import quat_mul

# Debug drawing
try:
    from omni.isaac.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except ImportError:
    pass

from tasks.push_t.pusht_env import compute_quat_angle
from tasks.ant_maze.antmaze_env_cfg import AntMazeEnvCfg
from tasks.ant_maze.gen_maze_states import MazeA, MazeB, MazeC, generate
from utils.misc import AverageScalarMeter, to_torch


class AntMazeEnv(DirectRLEnv):
    cfg: AntMazeEnvCfg

    def __init__(self, cfg: AntMazeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Flags to control reset behavior as the reset behavior needs
        # to alternate between planning and learning
        self.reset_dist_type = "eval"

        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robot.find_joints(".*")
        self.num_dofs = len(self._joint_dof_idx)
        print("***Debug: Number of DOFs: ", self.num_dofs, "***")

        # joint limits
        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.dof_pos_lower_lim = joint_pos_limits[..., 0][0]
        self.dof_pos_upper_lim = joint_pos_limits[..., 1][0]
        print("***Debug: Joint position lower limits: ", self.dof_pos_lower_lim, "***")
        print("***Debug: Joint position upper limits: ", self.dof_pos_upper_lim, "***")

        # Setup velocity limits
        self.dof_vel_lower_lim = torch.tensor(self.cfg.dof_vel_lower_lim, dtype=torch.float, device=self.device)
        self.dof_vel_upper_lim = torch.tensor(self.cfg.dof_vel_upper_lim, dtype=torch.float, device=self.device)

        # Setup torso velocity limits
        self.torso_vel_lower_lim = torch.tensor(self.cfg.torso_vel_lower_lim, dtype=torch.float, device=self.device)
        self.torso_vel_upper_lim = torch.tensor(self.cfg.torso_vel_upper_lim, dtype=torch.float, device=self.device)

        # Create Maze class object (to sample collision-free states in the PRM algorithm) and position limits
        if self.cfg.maze == "maze_a":
            self.maze_object = MazeA()
            self.torso_pos_upper_lim = torch.tensor([12.0, 12.0], device=self.device)
            self.torso_pos_lower_lim = torch.tensor([-12.0, -12.0], device=self.device)
        elif self.cfg.maze == "maze_b":
            self.maze_object = MazeB()
            self.torso_pos_upper_lim = torch.tensor([9.0, 9.0], device=self.device)
            self.torso_pos_lower_lim = torch.tensor([-9.0, -9.0], device=self.device)
        elif self.cfg.maze == "maze_c":
            self.maze_object = MazeC()
            self.torso_pos_upper_lim = torch.tensor([12.0, 12.0], device=self.device)
            self.torso_pos_lower_lim = torch.tensor([-12.0, -12.0], device=self.device)
        else:
            raise ValueError(f"Invalid maze name: {self.cfg.maze}")

        # create goal position
        self.goal = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)

        # Flags to control reset behavior as the reset behavior needs
        # to alternate between PRM and RL
        self.enable_reset = True
        self.reset_dist_type = "eval"

        # planning state (x) dimension
        self.planning_state_dim = 2 * self.num_dofs + 13  # joint pos, joint vel, torso state
        self.planner_goal_dim = 2  # 2 for goal xy coordinates

        # set up goal buffer from the sampled valid positions in maze
        possible_pos_xy = to_torch(generate(self.maze_object, 10000), device=self.device)
        self.goal_buf = possible_pos_xy[:, :2]

        # Logging success rate
        self.success = torch.zeros_like(self.reset_buf, dtype=torch.float)
        self.success_rate = AverageScalarMeter(100)
        self.extras["success_rate"] = 0.0

        # parameters for computing the delta reward
        self.pos_dist = torch.tensor(
            [torch.nan] * self.num_envs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(-1)
        self.prev_pos_dist = torch.tensor(
            [torch.nan] * self.num_envs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(-1)

        # Set up rotation axis for the object
        self.z_axis = torch.zeros((self.num_envs, 3), device=self.device)
        self.z_axis[:, 2] = 1

    def _setup_scene(self):
        """ Setup the scene with the robot, ground plane, and lights. """
        self.robot = Articulation(self.cfg.robot_cfg)
        self.maze = RigidObject(self.cfg.maze_cfg)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["maze"] = self.maze
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """ Pre-process actions before stepping through the physics. """
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """ Apply the actions to the robot based on target joint positions. """
        forces = self.action_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _get_observations(self) -> dict:
        """ Compute and return the observations for the environment.

        Returns:
            The observations (key: "policy") and states ((key: "critic") for the environment.
        """
        obs_policy = torch.cat(
            (
                self.torso_position,
                self.torso_rotation,
                self.torso_lin_vel,
                self.torso_ang_vel,
                self.joint_pos,
                self.joint_vel,
                self.goal,
            ),
            dim=1
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

        # Compute the distance between the torso position and the goal
        torso_pos_xy = self.torso_position[:, :2]
        self.pos_dist = torch.linalg.norm(torso_pos_xy - self.goal, dim=1).unsqueeze(-1)

        # compute delta reward
        nan_indices = torch.argwhere(torch.isnan(self.prev_pos_dist).float()).squeeze(-1)
        if len(nan_indices) > 0:
            self.prev_pos_dist[nan_indices] = self.pos_dist[nan_indices]

        # compute delta displacement
        delta_displacement = self.pos_dist - self.prev_pos_dist
        # delta reward
        reward = -1 * self.cfg.dense_reward_scale * delta_displacement
        reward = reward.squeeze(-1)
        self.prev_pos_dist[:] = self.pos_dist.clone()

        reward += (self.pos_dist.squeeze(-1) < self.cfg.at_goal_threshold) * self.cfg.success_reward_scale

        if self.reset_dist_type == "eval":
            self.success = torch.logical_or(self.success > 0, (self.pos_dist.squeeze() < self.cfg.at_goal_threshold))
            self.extras["success"] = self.success

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """ Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Shape of individual tensors is (num_envs,).
        """
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # reset if the torso is below the termination height
        done = self.torso_position[:, 2] < self.cfg.termination_height
        # reset if the projection of the up-direction on the z-axis is too small
        obj_axis = my_quat_rotate(self.torso_rotation, self.z_axis)
        done = torch.where(
            torch.linalg.norm(obj_axis * self.z_axis, dim=1) < self.cfg.torso_tilt_limit,
            torch.ones_like(self.reset_buf),
            done,
        )
        # reset if the robot finishes the task
        if self.cfg.reset_at_goal:
            done = torch.where(
                self.pos_dist.squeeze(-1) < self.cfg.at_goal_threshold,
                torch.ones_like(self.reset_buf),
                done
            )

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

        if self.reset_dist_type == "train":
            # sample random state from the reset distribution
            # print("Debug: reset state distribution size:", self.reset_state_buf.shape[0])
            sampled_idx = torch.randint(0, self.reset_state_buf.shape[0], (len(env_ids),))
            states = self.reset_state_buf[sampled_idx].to(self.device)
        else:
            joint_pos = self.robot.data.default_joint_pos[env_ids]
            joint_vel = self.robot.data.default_joint_vel[env_ids]
            default_ant_root_state = self.robot.data.default_root_state[env_ids]

            # reset the state of the environment
            states = torch.cat(
                (
                    joint_pos,
                    joint_vel,
                    default_ant_root_state
                ),
                dim=1
            )

        with torch.inference_mode():
            self.set_env_states(states, env_ids)
            self.simulate()

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

        # fetch root state from the robot
        self.torso_position = self.robot.data.root_pos_w - self.scene.env_origins[:, :3]
        self.torso_rotation = self.robot.data.root_quat_w
        self.torso_lin_vel, self.torso_ang_vel = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w

        # torso root state
        self.ant_root_state = torch.cat(
            [self.torso_position, self.torso_rotation, self.torso_lin_vel, self.torso_ang_vel], dim=1
        )

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

    def set_reset_dist_type(self, reset_dist_type: str) -> None:
        """ Set the reset distribution type. """
        self.reset_dist_type = reset_dist_type

    def save_episode_context(self):
        """ Saves episode context to switch to planner """
        context = {
            "progress_buf": self.episode_length_buf.detach().clone(),
            "reset_buf": self.reset_buf.detach().clone(),
            "dones": self.reset_buf.detach().clone(),
            "env_states": self.get_env_states(),
            "goal": self.goal.detach().clone() if hasattr(self, "goal") else "None",
            "prev_pos_dist": self.prev_pos_dist.detach().clone()
        }
        self.clear_prev_pos_dist()
        return context

    def restore_episode_context(self, context):
        """ Restore episode context from planning to learning """
        with torch.no_grad():
            self.progress_buf = context["progress_buf"]
            self.reset_buf = context["reset_buf"]
            self.dones = context["dones"]
            self.prev_pos_dist = context["prev_pos_dist"]
            with torch.inference_mode():
                self.set_env_states(context["env_states"], torch.arange(self.num_envs, device=self.device))
            if hasattr(self, "goal"):
                self.goal = context["goal"]
            self.get_observations()

        return self.get_observations()

    def clear_prev_pos_dist(self):
        self.prev_pos_dist = torch.tensor(
            [torch.nan] * self.num_envs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(-1)

    """ PRM Planner functions """

    def get_env_states(self) -> torch.Tensor:
        """ Returns the current state of the environment """
        # In this case, the state is the same as the q_state
        return torch.cat(
            [
                self.joint_pos.detach().clone(),
                self.joint_vel.detach().clone(),
                self.ant_root_state.detach().clone(),
            ],
            dim=1
        )

    def get_env_q(self) -> torch.Tensor:
        """ Returns the current q_state of the environment """
        return self.get_env_states()

    def set_env_states(self, states, env_ids: torch.Tensor) -> None:
        """ Sets the state of the envs specified by env_idx """
        # Extract joint position, velocity and root state from the states
        joint_pos = states[:, :self.num_dofs]
        joint_vel = states[:, self.num_dofs: 2 * self.num_dofs]
        ant_root_state = states[:, 2 * self.num_dofs:]

        ant_root_state[:, :3] = ant_root_state[:, :3] + self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(ant_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(ant_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def set_goal(self, goals: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.goal[env_ids, :] = goals

    def q_to_goal(self, q: torch.Tensor) -> torch.Tensor:
        """ Extract goal position from q_state """
        goal = q[:, 2 * self.num_dofs: 2 * self.num_dofs + 2]  # extract torso position
        return goal

    def compute_goal_distance(self, prm_nodes: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
            Computes distance in goal state from a specific node to each node in node set.
        """
        ant_torso_pos = prm_nodes[:, 2 * self.num_dofs: 2 * self.num_dofs + 2].view(-1,
                                                                                    2)  # extract ant torso positions from the nodes
        distances = torch.linalg.norm(ant_torso_pos - goal[2 * self.num_dofs: 2 * self.num_dofs + 2], dim=1)
        return distances

    def compute_distance(self, selected_node: torch.Tensor, prm_nodes: torch.Tensor) -> torch.Tensor:
        """ Computes distance from a specific node to each node in node set """
        # Ant torso position distance
        ant_torso_pos_dist = 1.0 * torch.linalg.norm(
            prm_nodes[:, 2 * self.num_dofs: 2 * self.num_dofs + 3] - selected_node[
                                                                     2 * self.num_dofs: 2 * self.num_dofs + 3], dim=1)
        # Ant torso rotation distance
        selected_node_quat = selected_node[2 * self.num_dofs + 3: 2 * self.num_dofs + 7].unsqueeze(0)
        nodes_quat = prm_nodes[:, 2 * self.num_dofs + 3: 2 * self.num_dofs + 7].view(-1, 4)
        ant_torso_rot_dist = 1.0 * compute_quat_angle(selected_node_quat, nodes_quat).squeeze()
        # Joint position distance
        joint_pos_dist = 0.1 * torch.linalg.norm(prm_nodes[:, 0: self.num_dofs] - selected_node[0: self.num_dofs],
                                                 dim=1)
        # Joint velocity distance
        joint_vel_dist = 0.01 * torch.linalg.norm(
            prm_nodes[:, self.num_dofs: 2 * self.num_dofs] - selected_node[self.num_dofs: 2 * self.num_dofs], dim=1)
        # Compute the total distance
        total_dist = ant_torso_pos_dist + ant_torso_rot_dist + joint_pos_dist + joint_vel_dist
        return total_dist

    def sample_q(self, num_samples) -> torch.Tensor:
        # Sample valid positions in maze
        init_states = generate(self.maze_object, num_samples)
        rand_torso_pos_xy = to_torch(init_states, device=self.device)[:, :2]
        # Sample random torso z coordinate
        alpha_torso_z = torch.rand((num_samples, 1), device=self.device)
        rand_torso_pos_z = alpha_torso_z * self.cfg.torso_z_upper_lim + (1 - alpha_torso_z) * self.cfg.torso_z_lower_lim
        # Sample random torso rotation within tilt limit
        rand_torso_rot = generate_quaternions_within_tilt(
            self.z_axis[0].cpu(),  # z-axis
            self.cfg.torso_tilt_limit,
            num_samples=num_samples,
        ).to(self.device)
        # Sample random joint position
        alpha_dof_pos = torch_rand_float(
            0.0,
            1.0,
            (num_samples, self.dof_pos_lower_lim.shape[0]),
            device=self.device,
        )
        rand_dof_pos = alpha_dof_pos * self.dof_pos_upper_lim + (1 - alpha_dof_pos) * self.dof_pos_lower_lim
        # Sample random joint velocity
        alpha_dof_vel = torch_rand_float(
            0.0,
            1.0,
            (num_samples, self.dof_vel_lower_lim.shape[0]),
            device=self.device,
        )
        rand_dof_vel = alpha_dof_vel * self.dof_vel_upper_lim + (1 - alpha_dof_vel) * self.dof_vel_lower_lim
        # Sample random torso velocity (linear + angular)
        alpha_torso_vel = torch_rand_float(
            0.0,
            1.0,
            (num_samples, self.torso_vel_lower_lim.shape[0]),
            device=self.device,
        )
        rand_torso_vel = alpha_torso_vel * self.torso_vel_upper_lim + (1 - alpha_torso_vel) * self.torso_vel_lower_lim

        # Concatenate all the states
        x_start = torch.cat(
            [
                rand_dof_pos,
                rand_dof_vel,
                rand_torso_pos_xy,
                rand_torso_pos_z,
                rand_torso_rot,
                rand_torso_vel
            ],
            dim=1
        )
        return x_start

    def sample_random_goal_state(self, num_goal) -> torch.Tensor:
        """ Sample goal positions which is close to the nodes in the node set """
        return self.sample_q(num_goal)

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

    def is_invalid(self, debug: bool = False) -> (torch.Tensor, torch.Tensor):
        """ Check if the sampled state is valid to be added to the graph """
        self._compute_intermediate_values()

        # z coordinate constraints
        invalid = self.torso_position[:, 2] < self.cfg.termination_height

        # xy coordinate constraints
        invalid_xy = torch.logical_or(
            torch.any(self.torso_position[:, :2] > self.torso_pos_upper_lim, dim=1),
            torch.any(self.torso_position[:, :2] < self.torso_pos_lower_lim, dim=1),
        )
        invalid = torch.logical_or(invalid, invalid_xy)

        obj_axis = my_quat_rotate(self.torso_rotation, self.z_axis)
        invalid = torch.where(
            torch.linalg.norm(obj_axis * self.z_axis, dim=1) < self.cfg.torso_tilt_limit,
            torch.ones_like(self.reset_buf),
            invalid,
        )

        x_start_prime = self.get_env_states()
        return invalid, x_start_prime


##
# torch.jit functions
##


@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def compute_up(torso_rotation, inv_start_rot, vec1, up_idx):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    num_envs = torso_rotation.shape[0]
    torso_quat = quat_mul(torso_rotation, inv_start_rot)
    up_vec = get_basis_vector(torso_quat, vec1).view(num_envs, 3)
    up_proj = up_vec[:, up_idx]
    return up_proj


def get_basis_vector(q, v):
    return quat_rotate(q, v)


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, 0]  # Take the w (scalar) component from the first position
    q_vec = q[:, 1:4]  # Take the x, y, z components from positions 1 to 3
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def my_quat_rotate(q, v):
    # Rotate along axis v by quaternion q
    shape = q.shape
    q_w = q[:, 0]  # scalar part (w)
    q_vec = q[:, 1:]  # vector part (x, y, z)
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


def generate_quaternions_within_tilt(axis, threshold_angle_radians, num_samples=1):
    """
    Generate random quaternions with the condition that the dot product of the rotated z-axis
    and the world z-axis is greater than cos(threshold_angle_radians).

    Parameters:
        threshold_angle_radians (float): The maximum allowed tilt angle (in radians).
        num_samples (int): The number of quaternions to generate.

    Returns:
        torch.Tensor: A tensor of shape (num_samples, 4) containing the quaternions [w, x, y, z].
    """
    valid_quaternions = []

    # Compute the cosine of the threshold angle to use for comparison
    cos_threshold = math.cos(threshold_angle_radians)

    while len(valid_quaternions) < num_samples:
        # Generate a random quaternion
        q = torch.rand((1, 4))
        q = q / torch.norm(q, dim=-1, keepdim=True)  # Normalize quaternion

        # Rotate the z-axis [0, 0, 1] by the quaternion
        v = axis.unsqueeze(0)  # z-axis vector
        v_rot = my_quat_rotate(q, v)  # Rotate the z-axis by the quaternion

        # Compute the dot product between the rotated z-axis and the world z-axis
        dot_product = v_rot[:, 2]  # Since the z-axis is [0, 0, 1], this is the z component of the rotated axis

        # Check if the dot product is greater than or equal to the cosine of the threshold angle
        if dot_product >= cos_threshold:
            valid_quaternions.append(q.squeeze(0))

    return torch.stack(valid_quaternions)


##
# Debug drawing
##


def draw_plus(points, color="r"):
    """ Draw the goal position as a red cross in the maze """
    cmap = {"r": [1, 0, 0, 1], "g": [0, 1, 0, 1], "b": [0, 0, 1, 1]}
    color_rgba = cmap[color]
    for point in points:
        point_list_0 = [(point[0] - 0.3, point[1], 0.1), (point[0], point[1] - 0.3, 0.1)]
        point_list_1 = [(point[0] + 0.3, point[1], 0.1), (point[0], point[1] + 0.3, 0.1)]
        colors = [color_rgba for _ in range(2)]
        widths = [1.0 for _ in range(2)]
        try:
            draw.draw_lines(point_list_0, point_list_1, colors, widths)
        except:
            pass