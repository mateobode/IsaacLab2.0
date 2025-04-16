# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg


WALLS_CFG = RigidObjectCollectionCfg(
    rigid_objects={
        f"Wall_{i}": RigidObjectCfg(
            prim_path = f"/World/envs/env_.*/Wall_{i}",
            spawn = sim_utils.CuboidCfg(
                size=(2.0, 0.1, 0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=False,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=100.0,
                    enable_gyroscopic_forces=True,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(i, 0.0, 0.25)),  # Default position
        )
        for i in range(20)  # Create 10 walls at different positions
    }
)

WALL_CFG = RigidObjectCollectionCfg(
    rigid_objects={
        f"Wall_{i}": RigidObjectCfg(
            prim_path = f"/World/envs/env_.*/Wall_{i}",
            spawn = sim_utils.CuboidCfg(
                size=(2.0, 0.1, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4, 
                    solver_velocity_iteration_count=0,
                    kinematic_enabled=True,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),  # Default position
        )
        for i in range(20)  # Create 10 cones at different positions
    }
)