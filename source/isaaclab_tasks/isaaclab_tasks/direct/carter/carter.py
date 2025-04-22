# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Carter V1 robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration - Actuators
##

CARTER_V1_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["left_wheel", "right_wheel"],
    #effort_limit=40000.0,
    effort_limit_sim=100.0,
    #velocity_limit=100.0,
    stiffness=0.0,
    #damping=10.0,
    damping=1.0e3
)

#CASTER_CFG = ImplicitActuatorCfg(
#    joint_names_expr=["rear_pivot", "rear_axle"],
#    effort_limit=0.0,
#    velocity_limit=0.0,
#    stiffness=0.0,
#    damping=0.0,
#)

##
# Configuration - Articulation
##

CARTER_V1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Carter/carter_v1_physx_lidar.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #disable_gravity=False,
            rigid_body_enabled=True,
            #retain_accelerations=False,
            linear_damping=0.1,
            angular_damping=0.1,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_wheel": 0.0,
            "right_wheel": 0.0,
            "rear_pivot": 0.0,
            "rear_axle": 0.0
        },
    ),
    actuators={
        "wheels": CARTER_V1_ACTUATOR_CFG,
        #"caster": CASTER_CFG,
    },
)

"""Configuration for the Carter V1 robot."""