from omni.isaac.lab.app import AppLauncher
import hydra
from omegaconf import DictConfig
from utils.misc import omegaconf_to_dict

# Declare the global variable
simulation_app = None

@hydra.main(config_name="test_prm_pushmaze", config_path="cfg", version_base="1.2")
def create_sim_app(cfg: DictConfig):
    global simulation_app

    # parse the config file and convert it to a dictionary
    maze_cfg = omegaconf_to_dict(cfg)
    app_launcher = AppLauncher(headless=maze_cfg["headless"])
    simulation_app = app_launcher.app

# launch omniverse app
create_sim_app()



import os
from time import sleep
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.utils import update_class_from_dict

from utils.misc import set_np_formatting, set_seed
from tasks.push_maze.config.maze import MAZEA_CFG, MAZEB_CFG, MAZEC_CFG


@hydra.main(config_name="test_prm_pushmaze", config_path="cfg", version_base="1.2")
def build_prm_mazebot(cfg: DictConfig):
    """ Test the MazeBot task with random actions """
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
    env_cfg.robot_cfg = MAZE_CFG[selected_maze].replace(prim_path="/World/envs/env_.*/Robot")

    # create DirectRLEnv
    env = gym.make(maze_cfg["task_id"], cfg=env_cfg)

    # create sampling-based planner
    planner_cfg = maze_cfg["planner"]
    planner = hydra.utils.get_class(planner_cfg["_target_"])(
        cfg=planner_cfg,
        env=env,
        buffer=None,
        actor_target=None,
        critic_target=None,
        obs_policy_rms=None,
        obs_critic_rms=None,
        value_rms=None,
        device=maze_cfg["rl_device"],
        gamma=0.99,
    )

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    if maze_cfg["task_type"] == "grow":
        # run PRM
        last_value = 0
        planning_steps = 0
        epochs = []
        num_nodes_list = []
        while True:
            planner.run_prm()
            planning_steps += 1
            # save the prm every 1000 nodes
            num_nodes = planner.prm_q.shape[0]
            # record the epoch and nodes number
            epochs.append(planning_steps)
            num_nodes_list.append(planner.prm_q.shape[0])
            os.makedirs(maze_cfg["saved_file_name_format"], exist_ok=True)
            if num_nodes // 1000 != last_value:
                last_value += 1
                planner.save_prm(f"{maze_cfg['saved_file_name_format']}_{num_nodes}.pkl")
                # save nodes number data
                if maze_cfg["save_num_nodes"]:
                    np.save(
                        f"epoch_{maze_cfg['planner']['new_state_portion']}_{num_nodes}.npy",
                        np.array(epochs)
                    )
                    np.save(
                        f"num_nodes_list_{maze_cfg['planner']['new_state_portion']}_{num_nodes}.npy",
                        np.array(num_nodes_list)
                    )
    elif maze_cfg["task_type"] == "visualize":
        planner.load_prm(maze_cfg["saved_prm_file"])
        print("Average children number in the graph: ", planner.children_counter.float().mean())
    else:
        raise ValueError("Invalid task type")

    # visualize the demo
    while simulation_app.is_running():
        print("***** Extract demos from PRM *****")
        # Extract demonstrations
        obs_policy_demo, *_ = planner.extract_demos(num_demos=1, max_len=10, num_parents=1)
        env.set_env_states_from_obs(obs_policy_demo[:, 0, :])
        sleep(2.0)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    build_prm_mazebot()
    # close sim app
    simulation_app.close()