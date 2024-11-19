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
    repo_name = "DPRM"
    repo_root = fetch_repo_root(current_file_path, repo_name)
    sys.path.append(repo_root)
    print(f"Repository root '{repo_root}' added to Python path.")
except ValueError as e:
    print(e)

from omni.isaac.lab.app import AppLauncher
import hydra
from omegaconf import DictConfig
from utils.misc import omegaconf_to_dict

# Declare the global variable
simulation_app = None

@hydra.main(config_name="train_antmaze", config_path="cfg", version_base="1.2")
def create_sim_app(cfg: DictConfig):
    global simulation_app

    # parse the config file and convert it to a dictionary
    task_cfg = omegaconf_to_dict(cfg)
    app_launcher = AppLauncher(headless=task_cfg["headless"])
    simulation_app = app_launcher.app

# launch omniverse app
create_sim_app()


import gymnasium as gym
import wandb
from hydra.core.hydra_config import HydraConfig

from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.utils import update_class_from_dict

from algo.diffusion_roadmap import DiffusionRoadmap
from algo.dppo_roadmap import DPPORoadmap
from utils.misc import set_np_formatting, set_seed
from tasks.ant_maze.antmaze_env_cfg import AntMazeEnvCfg
from tasks.ant_maze.config.maze import MAZEA_CFG, MAZEB_CFG, MAZEC_CFG

@hydra.main(config_name="train_antmaze", config_path="cfg", version_base="1.2")
def build_learning_env(cfg: DictConfig):
    """ Create the environment and run the planning and learning algorithms """
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

    output_dir = HydraConfig.get().runtime.output_dir
    # agent = DiffusionRoadmap(cfg=maze_cfg, env=env, output_dir=output_dir)
    agent = DPPORoadmap(cfg=maze_cfg, env=env, output_dir=output_dir)

    wandb.init(
        project="AntMaze",  # set the wandb project where this run will be logged
        # config=OmegaConf.to_container(cfg, resolve=False),  # track hyperparameters and run metadata
        name=output_dir,
    )

    if maze_cfg["test"]:
        agent.restore_test(maze_cfg["load_path"])
        agent.test()
    else:
        agent.train()

    # close sim app
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    build_learning_env()
    # close sim app
    simulation_app.close()