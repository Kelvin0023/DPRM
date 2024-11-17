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


import gymnasium as gym
from tasks.maze.mazebot_env_cfg import MazeBotEnvCfg


##
# Register Gym environments.
##


gym.register(
    id="Isaac-MazeBot-v0",
    entry_point="tasks.maze.mazebot_env:MazeBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MazeBotEnvCfg,
    },
)