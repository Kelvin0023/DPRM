import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils

from tasks.pushmaze_3d.config.ur5e import UR5E_PROBE_CFG


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
    reset_scene_position = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class PushMaze3DEnvCfg(DirectRLEnvCfg):
    """ Configuration for the Push-T environment. """
    """ The configuration below would be overwritten by the configuration in the task yaml config file. """

    # env
    decimation = 2
    episode_length_s = 10.0
    num_actions = 6
    num_observations = 29
    num_states = 29
    num_q_space = 23
    max_episode_steps = 600

    # PRM config for observation update
    extracted_goal_idx_policy = (12, 18)  # indices of the current observation (obj_pose)
    goal_idx_policy = (23, 29)  # indices of the goal observation (goal_pose)
    extracted_goal_idx_critic = (12, 18)  # indices of the current critic observation (obj_pose)
    goal_idx_critic = (23, 29)  # indices of the goal critic observation (goal_pose)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
    )

    # UR5e robot arm
    robot_cfg: ArticulationCfg = UR5E_PROBE_CFG.replace(prim_path="/World/envs/env_.*/robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        )
    )
    actuated_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    probe_body_name = ["probe_link"]

    # object to push
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(repo_root, "tasks/pushmaze_3d/assets/object/usd/cube_2cm_usd/cube_2cm.usd"),
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
            mass_props=sim_utils.MassPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3305), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=os.path.join(repo_root, "tasks/pushmaze_3d/assets/object/usd/cube_2cm_usd/cube_2cm.usd"),
                scale=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),
            ),
        },
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # PRM valid state limits
    valid_probe_object_thres = 0.2
    max_height = 0.3305
    min_height = 0.3295

    # robot arm joint limits
    dof_pos_lower_limit = [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14]
    dof_pos_upper_limit = [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
    dof_vel_lower_limit = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    dof_vel_upper_limit = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # object limits
    obj_pos_lower_limit = [-0.3, -0.3]
    obj_pos_upper_limit = [0.3, 0.3]
    obj_vel_lower_limit = [-1.0, -1.0]
    obj_vel_upper_limit = [1.0, 1.0]

    # default hand and object configuration
    default_hand_joint_pos = [0, -1.57, 1.57, -1.57, -1.57, 0]
    default_object_pos = [0.0, 0.0, 0.3305]
    default_object_quat = [1.0, 0.0, 0.0, 0.0]

    # action scale
    action_scale = 0.1

    # reward and termination
    reward_type = "mixed"  # sparse, dense, or mixed
    pos_dense_reward_scale = 0.01
    rot_dense_reward_scale = 0.01
    success_reward_scale = 1.0  # sparse reward for pushing object to the goal position

    success_pos_threshold = 0.1  # distance to goal to consider as success
    success_rot_threshold = 0.3  # rotation to goal to consider as success


    # domain reset and randomization config
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # set up the viewer
        self.viewer.eye = (0.0, 0.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)




