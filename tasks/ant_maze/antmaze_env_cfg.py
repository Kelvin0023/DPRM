import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from tasks.ant_maze.config.maze import MAZEA_CFG, MAZEB_CFG, MAZEC_CFG
from tasks.ant_maze.config.ant import ANT_CFG

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
    repo_name = "diffuse-plan-learn"
    repo_root = fetch_repo_root(current_file_path, repo_name)
    sys.path.append(repo_root)
    print(f"Repository root '{repo_root}' added to Python path.")
except ValueError as e:
    print(e)


@configclass
class EventCfg:
    """Configuration for environment reset and randomization."""
    # -- robot rigid body properties -- #
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    # -- maze rigid body properties -- #
    maze_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("maze", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    # reset
    reset_maze_position = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

@configclass
class AntMazeEnvCfg(DirectRLEnvCfg):
    """ Configuration for the Ant Maze environment """

    decimation = 1
    episode_length_s = 10.0
    action_scale = 0.5
    num_actions = 8
    num_observations = 31
    num_states = 31
    num_q_space = 4
    max_episode_steps = 600

    # maze and robot
    maze = "maze_a"

    MAZE_CFG = {
        "maze_a": MAZEA_CFG,
        "maze_b": MAZEB_CFG,
        "maze_c": MAZEC_CFG,
    }

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        # physics_material=sim_utils.RigidBodyMaterialCfg(
        #     friction_combine_mode="average",  # Defaults to 'average'
        #     restitution_combine_mode="average",  # Defaults to 'average'
        #     static_friction=5.0,  # Defaults to 0.5
        #     dynamic_friction=1.0,  # Defaults to 0.5
        #     restitution=0.0,  # Defaults to 0.0
        # ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=60.0, replicate_physics=True)

    maze_cfg: RigidObjectCfg = MAZE_CFG[maze].replace(prim_path="/World/envs/env_.*/Maze", collision_group=0)

    # robot
    robot_cfg: ArticulationCfg = ANT_CFG.replace(prim_path="/World/envs/env_.*/Robot", collision_group=0).replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={
                ".*_leg": 0.0,
                "front_left_foot": 0.785398,  # 45 degrees
                "front_right_foot": -0.785398,
                "left_back_foot": -0.785398,
                "right_back_foot": 0.785398,
            },
        )
    )

    # domain reset and randomization config
    events: EventCfg = EventCfg()

    """Configuration for the Mujoco Ant robot."""
    joint_gears: list = [15, 15, 15, 15, 15, 15, 15, 15]


    # reward
    dense_reward_scale = 1.0
    success_reward_scale = 10.0
    at_goal_threshold = 0.2
    reset_at_goal = True

    termination_height = 0.31
    torso_tilt_limit = 0.7

    # PRM sampling space limits
    torso_z_lower_lim = 0.4
    torso_z_upper_lim = 0.6
    torso_vel_lower_lim = [-0.5, -0.5, -0.1, -0.5, -0.5, -0.8]
    torso_vel_upper_lim = [0.5, 0.5, 0.1, 0.5, 0.5, 0.8]
    dof_vel_lower_lim = [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]
    dof_vel_upper_lim = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    # PRM config for observation update
    extracted_goal_idx_policy = (0, 2)  # indices of the current observation (torso_pos_xy)
    goal_idx_policy = (29, 31)  # indices of the goal observation (goal_pos_xy)
    extracted_goal_idx_critic = (0, 2)  # indices of the current critic observation (torso_pos_xy)
    goal_idx_critic = (29, 31)  # indices of the goal critic observation (goal_pos_xy)

    def __post_init__(self) -> None:
        """Post initialization."""
        # set up the viewer
        self.viewer.eye = (0.0, 0.0, 100.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

