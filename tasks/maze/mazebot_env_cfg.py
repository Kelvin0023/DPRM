import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass

from tasks.maze.config.maze import MAZEA_CFG, MAZEB_CFG, MAZEC_CFG


@configclass
class EventCfg:
    """Configuration for environment reset and randomization."""
    # reset
    reset_maze_position = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class MazeBotEnvCfg(DirectRLEnvCfg):
    """ Configuration for the MazeBot environment. """
    """ The configuration below would be overwritten by the configuration in the task yaml config file. """

    # env
    decimation = 2
    episode_length_s = 10.0
    num_actions = 2
    num_observations = 6
    num_states = 6
    num_q_space = 4
    max_episode_steps = 600

    # PRM config for observation update
    extracted_goal_idx_policy = (0, 2)  # indices of the current observation (robot_pos)
    goal_idx_policy = (4, 6)  # indices of the goal observation (goal_pos)
    extracted_goal_idx_critic = (0, 2)  # indices of the current critic observation (robot_pos)
    goal_idx_critic = (4, 6)  # indices of the goal critic observation (goal_pos)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
    )

    # maze and robot
    maze = "maze_a"

    MAZE_CFG = {
        "maze_a": MAZEA_CFG,
        "maze_b": MAZEB_CFG,
        "maze_c": MAZEC_CFG,
    }
    robot_cfg: ArticulationCfg = MAZE_CFG[maze].replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        )
    )
    actuated_joint_names = [
        "base_to_bot_x",
        "bot_x_to_bot",
    ]

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # velocity limit
    dof_vel_lower_limit = [-1.0, -1.0]
    dof_vel_upper_limit = [1.0, 1.0]

    # PD controller setting
    kp = 5
    kd = 2

    # reward and termination
    reward_type = "dense"  # dense or sparse
    dense_reward_scale = 1.0  # dense reward for moving towards the goal position
    success_reward_scale = 1.0  # sparse reward for reaching the goal position
    at_goal_threshold = 0.1  # distance to goal to consider as success

    # domain reset and randomization config
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # set up the viewer
        self.viewer.eye = (0.0, 0.0, 4)
        self.viewer.lookat = (0.0, 0.0, 0.0)




