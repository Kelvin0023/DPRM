import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils

from tasks.push_t.config.robot import ROBOT_2D_CFG


# Fetch the repository root
import os
import sys

def fetch_repo_root(file_path, repo_name):
    # Split the file path into parts
    path_parts = file_path.split(os.sep)

    # Try to find the repository name in the path
    if repo_name in path_parts:
        # Find the index of the repository name
        repo_index = path_parts.index(repo_name)
        # Join the path components up to the repository name
        repo_root = os.sep.join(path_parts[:repo_index + 1])
        return repo_root
    else:
        raise ValueError("Repository name not found in the file path")

try:
    current_file_path = os.path.abspath(__file__)
    repo_name = "DPRM"
    repo_root = fetch_repo_root(current_file_path, repo_name)
    sys.path.append(repo_root)
    print(f"Repository root '{repo_root}' added to Python path.")
except ValueError as e:
    print(e)


@configclass
class EventCfg:
    """Configuration for environment reset and randomization."""
    # reset
    reset_maze_position = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class PushTEnvCfg(DirectRLEnvCfg):
    """ Configuration for the Push-T environment. """
    """ The configuration below would be overwritten by the configuration in the task yaml config file. """

    # env
    decimation = 2
    episode_length_s = 10.0
    num_actions = 2
    num_observations = 21
    num_states = 21
    num_q_space = 15
    max_episode_steps = 600

    # PRM config for observation update
    extracted_goal_idx_policy = (4, 10)  # indices of the current observation (obj_pose)
    goal_idx_policy = (15, 21)  # indices of the goal observation (goal_pose)
    extracted_goal_idx_critic = (4, 10)  # indices of the current critic observation (obj_pose)
    goal_idx_critic = (15, 21)  # indices of the goal critic observation (goal_pose)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
    )

    # 2D navigation robot config
    robot_cfg: ArticulationCfg = ROBOT_2D_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "base_to_bot_x": 0.0,
                "bot_x_to_bot": 0.0,
            },
        )
    )
    actuated_joint_names = [
        "base_to_bot_x",
        "bot_x_to_bot",
    ]

    # object to push
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(repo_root, "tasks/push_t/assets/object/usd/t_block_usd/t_block.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.2,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.5, 0.1505), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=os.path.join(repo_root, "tasks/push_t/assets/object/usd/t_block_usd/t_block.usd"),
                scale=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, -0.5, 0.1505)),
            ),
        },
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=6.0, replicate_physics=True)

    # PRM valid state limits
    valid_robot_object_thres = 1.0
    height_limit = 0.1505
    dof_pos_lower_limit = [-1.2, -1.2]
    dof_pos_upper_limit = [1.2, 1.2]
    dof_vel_lower_limit = [-1.0, -1.0]
    dof_vel_upper_limit = [1.0, 1.0]
    obj_pos_lower_limit = [-0.5, -0.5]
    obj_pos_upper_limit = [0.5, 0.5]
    obj_vel_lower_limit = [-1.0, -1.0]
    obj_vel_upper_limit = [1.0, 1.0]

    # PD controller setting
    kp = 5
    kd = 2

    # reward and termination
    reward_type = "sparse"  # sparse, dense, or mixed
    pos_dense_reward_scale = 1.0
    rot_dense_reward_scale = 1.0
    success_reward_scale = 10.0  # sparse reward for pushing object to the goal position

    success_pos_threshold = 0.1  # distance to goal to consider as success
    success_rot_threshold = 0.3  # rotation to goal to consider as success


    # domain reset and randomization config
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # set up the viewer
        self.viewer.eye = (0.0, 0.0, 4)
        self.viewer.lookat = (0.0, 0.0, 0.0)




