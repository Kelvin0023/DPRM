# Add the repository root to the python path
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


from omni.isaac.lab.app import AppLauncher
import hydra
from omegaconf import DictConfig
from utils.misc import omegaconf_to_dict


# Create the global variable to store the simulation app
simulation_app = None

@hydra.main(config_name="test_maze", config_path="../../cfg", version_base="1.2")
def create_sim_app(cfg: DictConfig):
    global simulation_app

    # parse the config file and convert it to a dictionary
    task_cfg = omegaconf_to_dict(cfg)
    app_launcher = AppLauncher(headless=task_cfg["headless"])
    simulation_app = app_launcher.app

# launch omniverse app
create_sim_app()


import gymnasium as gym
import torch

from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.utils import update_class_from_dict

from utils.misc import set_np_formatting, set_seed
from tasks.ant_maze.antmaze_env_cfg import AntMazeEnvCfg
from tasks.ant_maze.config.maze import MAZEA_CFG, MAZEB_CFG, MAZEC_CFG


@hydra.main(config_name="test_antmaze", config_path="../../cfg", version_base="1.2")
def main(cfg: DictConfig):
    """ Test the MazeBot task with random actions """
    # declare the global variable
    global simulation_app

    # set numpy formatting for printing only
    set_np_formatting()

    # set seed
    cfg.seed = set_seed(cfg.seed)

    # parse the config file and convert it to a dictionary
    maze_cfg = omegaconf_to_dict(cfg)

    # create environment configuration
    env_cfg = parse_env_cfg(
        maze_cfg["task_id"],
        use_gpu=True if maze_cfg["pipeline"] == "gpu" else False,
        num_envs=maze_cfg["num_envs"],
        use_fabric=not maze_cfg["disable_fabric"],
    )

    # update the DirectRLEnvCfg with the task configuration
    update_class_from_dict(env_cfg, maze_cfg["task"])

    # select the maze configuration and override the robot configuration
    selected_maze = maze_cfg["task"]["maze"]
    MAZE_CFG = {
        "maze_a": MAZEA_CFG,
        "maze_b": MAZEB_CFG,
        "maze_c": MAZEC_CFG,
    }
    env_cfg.maze_cfg = MAZE_CFG[selected_maze].replace(prim_path="/World/envs/env_.*/Maze")

    # create DirectRLEnv
    env = gym.make(maze_cfg["task_id"], cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from normal distribution N(0, 1)
            actions = env.random_actions()
            # action clipping
            actions = torch.clamp(actions, -1.0, 1.0)
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
