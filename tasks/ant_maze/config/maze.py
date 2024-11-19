# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Fetch the repository root
import os

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
except ValueError as e:
    print(e)

"""Configuration for the maze in AntMaze task."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg

MAZE_USD_PATH = {
    "A": os.path.join(repo_root, "tasks/ant_maze/assets/maze/usd/maze_a/maze_a.usd"),
    "B": os.path.join(repo_root, "tasks/ant_maze/assets/maze/usd/maze_b/maze_b.usd"),
    "C": os.path.join(repo_root, "tasks/ant_maze/assets/maze/usd/maze_c/maze_c.usd"),
}

##
# Configuration
##

class MazeConfig:
    @staticmethod
    def create_config(usd_path: str) -> RigidObjectCfg:
        return RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=10000.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

MAZEA_CFG = MazeConfig.create_config(MAZE_USD_PATH["A"])
MAZEB_CFG = MazeConfig.create_config(MAZE_USD_PATH["B"])
MAZEC_CFG = MazeConfig.create_config(MAZE_USD_PATH["C"])

"""Configuration for the robot in MazeBot."""