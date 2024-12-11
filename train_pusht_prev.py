from omni.isaac.lab.app import AppLauncher
import hydra
from omegaconf import DictConfig
from utils.misc import omegaconf_to_dict

# Declare the global variable
simulation_app = None

@hydra.main(config_name="train_pusht_prev", config_path="cfg", version_base="1.2")
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

from algo.diff_roadmap import DiffusionRoadmap
from utils.misc import set_np_formatting, set_seed
from tasks.push_t.pusht_env_cfg import PushTEnvCfg


@hydra.main(config_name="train_pusht_prev", config_path="cfg", version_base="1.2")
def build_learning_env(cfg: DictConfig):
    """ Create the environment and run the planning and learning algorithms """
    global simulation_app

    # set numpy formatting for printing only
    set_np_formatting()

    # set seed
    cfg.seed = set_seed(cfg.seed)

    # parse the config file and convert it to a dictionary
    task_cfg = omegaconf_to_dict(cfg)

    # create environment configuration
    env_cfg = parse_env_cfg(
        task_cfg["task_id"],
        use_gpu=True if task_cfg["pipeline"] == "gpu" else False,
        num_envs=task_cfg["num_envs"],
        use_fabric=not task_cfg["disable_fabric"],
    )

    # update the DirectRLEnvCfg with the task configuration
    update_class_from_dict(env_cfg, task_cfg["task"])
    # create DirectRLEnv
    print("task id: ", task_cfg["task_id"])
    env = gym.make(task_cfg["task_id"], cfg=env_cfg)

    output_dir = HydraConfig.get().runtime.output_dir
    agent = DiffusionRoadmap(cfg=task_cfg, env=env, output_dir=output_dir)

    wandb.init(
        project="PushT_prev",  # set the wandb project where this run will be logged
        # config=OmegaConf.to_container(cfg, resolve=False),  # track hyperparameters and run metadata
        name=output_dir,
    )

    if task_cfg["test"]:
        agent.restore_test(task_cfg["load_path"])
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