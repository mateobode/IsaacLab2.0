# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg


"""Configuration for the Goals waypoint """


WAYPOINT_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/Cones",
    markers={
        "marker_0": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)
            )
        ),
        "marker_1": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0)
            )
        ),
    }
)

GOAL_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/Cones",
    markers={
        # region Red Markers
        "marker0": sim_utils.SphereCfg(
            radius=-0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker1": sim_utils.SphereCfg(
            radius=-0.09,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker2": sim_utils.SphereCfg(
            radius=-0.08,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker3": sim_utils.SphereCfg(
            radius=-0.07,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker4": sim_utils.SphereCfg(
            radius=-0.06,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker5": sim_utils.SphereCfg(
            radius=-0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker6": sim_utils.SphereCfg(
            radius=-0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker7": sim_utils.SphereCfg(
            radius=-0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker8": sim_utils.SphereCfg(
            radius=-0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker9": sim_utils.SphereCfg(
            radius=-0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        # region Blue Markers
        "marker10": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker11": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker12": sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker13": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker14": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker15": sim_utils.SphereCfg(
            radius=0.06,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker16": sim_utils.SphereCfg(
            radius=0.07,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker17": sim_utils.SphereCfg(
            radius=0.08,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker18": sim_utils.SphereCfg(
            radius=0.09,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker19": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    }
)