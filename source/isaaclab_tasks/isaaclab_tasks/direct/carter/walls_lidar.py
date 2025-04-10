# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

WALL_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(1.0, 0.1, 0.5),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    )
)

"""
WALL_CFG = RigidObjectCollectionCfg(
    rigid_objects={
        **{f"left_wall_{i}": RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/left_wall_{i}",
            spawn=sim_utils.CuboidCfg(
                size=(2.0, 0.1, 0.5),  # Longer walls, fewer segments needed
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.25)
            )
        ) for i in range(9)},
        
        **{f"right_wall_{i}": RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/right_wall_{i}",
            spawn=sim_utils.CuboidCfg(
                size=(2.0, 0.1,.5),  # Longer walls, fewer segments needed
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=False
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.25)
            )
        ) for i in range(9)}
    }
)
"""