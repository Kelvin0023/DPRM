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

"""Configuration for the UR5e hand with a probe."""
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

ur5e_probe_usd = os.path.join(repo_root, "tasks/pushmaze_3d/assets/robot/usd/ur5e_mazeA/ur5e_probe_mazeA.usd")

##
# Configuration
##

UR5E_PROBE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ur5e_probe_usd,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.57,
            "elbow_joint": 1.57,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "robot": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=3.0,
            damping=0.5,
        ),
    },
)

"""Configuration for the UR5e hand with a probe"""
