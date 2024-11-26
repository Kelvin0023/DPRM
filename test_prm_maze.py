from omni.isaac.lab.app import AppLauncher
import hydra
from omegaconf import DictConfig
from utils.misc import omegaconf_to_dict

# Declare the global variable
simulation_app = None

@hydra.main(config_name="test_prm_maze", config_path="cfg", version_base="1.2")
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
import numpy as np
import matplotlib.pyplot as plt

from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.utils import update_class_from_dict

from utils.misc import set_np_formatting, set_seed
from tasks.maze.config.maze import MAZEA_CFG, MAZEB_CFG, MAZEC_CFG


@hydra.main(config_name="test_prm_maze", config_path="cfg", version_base="1.2")
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
        model_target=None,
        obs_policy_rms=None,
        obs_critic_rms=None,
        value_rms=None,
        device=maze_cfg["rl_device"],
    )

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    # run PRM
    epochs = []
    num_nodes_list = []
    for epoch in range(200):
        planner.run_prm()
        # Store the values for plotting
        epochs.append(epoch)
        num_nodes_list.append(planner.prm_q.shape[0])

    # save data
    if maze_cfg["save_num_nodes"]:
        np.save(f"epoch_{maze_cfg['planner']['new_state_portion']}.npy", np.array(epochs))
        np.save(f"num_nodes_list_{maze_cfg['planner']['new_state_portion']}.npy", np.array(num_nodes_list))

    # Plot num_nodes vs. epoch
    plt.plot(epochs, num_nodes_list, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Nodes')
    plt.title('Number of Nodes vs Epoch')
    plt.grid(True)
    plt.show()

    # Perform random walk
    # walk, obs_buf, act_buf = planner.extract_walks(num_walks=5, length=10)
    # print("***** Extract Random Walks on PRM *****")
    # for i in range(walk.shape[0]):
    #     for j in range(walk.shape[1] - 1):
    #         if walk[i, j + 1, 0] == float('-inf'):
    #             break
    #         planner._visualize_new_edges(
    #             walk[i, j, :].unsqueeze(0),
    #             walk[i, j + 1, :].unsqueeze(0),
    #             edge_color=[0, 0, 0, 1],
    #             node_color=[1, 0, 0, 1]
    #         )
    # print("***** End of Random Walks *****")

    planner.extract_demos(num_demos=2, max_len=20, num_parents=3)

    # simulate environment with zero actions
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # take zero actions
            zero_action = env.zero_actions()
            # apply actions
            env.step(zero_action)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    build_prm_mazebot()
    # close sim app
    simulation_app.close()