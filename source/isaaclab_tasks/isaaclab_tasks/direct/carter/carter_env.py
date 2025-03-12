# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

from .carter import CARTER_V1_CFG
from .goal import CONE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg



@configclass
class CarterEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2 # Decimation factor for rendering
    episode_length_s = 10.0 # Maximum episode length in seconds
    action_space = 1 # Number of actions the neural network should return (wheel velocities)
    # Number of observations fed to the neural network
    observation_space = 1
    state_space = 0

    env_spacing = 5.0 # Spacing between environments

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CARTER_V1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    wheels = ["left_wheel", "right_wheel"]
    caster = ["rear_pivot", "rear_axle"]

    # goal
    cone_cfg: VisualizationMarkersCfg = CONE_CFG

    # lidar
    lidar: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/chassis_link/carter_lidar",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.5),
        ),
        mesh_prim_paths=["/World/ground"],
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=100,
            vertical_fov_range=[-90, 90],
            horizontal_fov_range=[-180, 180],
            horizontal_res=1.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

class CarterEnv(DirectRLEnv):
    cfg: CarterEnvCfg

    def __init__(self, cfg: CarterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._wheels_idx, _ = self.carter.find_joints(self.cfg.wheels)
        self._caster_idx, _ = self.carter.find_joints(self.cfg.caster)

        self.num_goals = 1

        self._goal_position = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._marker_position = torch.zeros((self.num_envs, self.num_goals, 3), dtype=torch.float32, device=self.device)

    def _setup_scene(self):
        self.carter = Articulation(self.cfg.robot_cfg)
        
        # add lidar
        self.lidar = RayCaster(self.cfg.lidar)
        
        # add goal marker
        self.goal_marker = VisualizationMarkers(self.cfg.cone_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False) # Clone child environments from parent environment
        self.scene.filter_collisions(global_prim_paths=[]) # Prevent environments from colliding with each other

        # add articulation to scene
        self.scene.articulations["carter"] = self.carter

        # add lidar to scene
        self.scene.sensors["lidar"] = self.lidar

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        action_scale = 10.0
        
        self.actions = torch.abs(action_scale * actions.reshape(-1, 1).repeat(1, 2))
        self._caster_action = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
                
    def _apply_action(self) -> None:
        self.carter.set_joint_velocity_target(self.actions, joint_ids=self._wheels_idx)
        self.carter.set_joint_position_target(self._caster_action, joint_ids=self._caster_idx)

    def _get_observations(self) -> dict:
        obs = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        failure_termination = self.episode_length_buf >= self.max_episode_length - 1
        clean_termination = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        return failure_termination, clean_termination

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.carter._ALL_INDICES
        super()._reset_idx(env_ids)
        
        num_reset = len(env_ids)

        # Reset from config
        default_state = self.carter.data.default_root_state[env_ids]
        carter_pose =  default_state[env_ids, :7]
        carter_velocities = default_state[env_ids, 7:]
        joint_positions = self.carter.data.default_joint_pos[env_ids]
        joint_velocities = self.carter.data.default_joint_vel[env_ids]

        carter_pose[:, :3] = self.scene.env_origins[env_ids]

        # Randomize starting position
        carter_pose[:, :2] += 2.0 * torch.rand((num_reset, 2), device=self.device)

        # Randomize starting heading
        angles = torch.pi * torch.rand((num_reset,), dtype=torch.float32, device=self.device)

        # Isaac Sim uses quaternions for rotations, quaternions are W-first (W, X, Y, Z)
        # To rotate about Z-axis, we need to modify values in W and Z
        carter_pose[:, 3] = torch.cos(angles / 0.5)
        carter_pose[:, 6] = torch.sin(angles / 0.5)

        self.carter.write_root_pose_to_sim(carter_pose, env_ids)
        self.carter.write_root_velocity_to_sim(carter_velocities, env_ids)
        self.carter.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)

        
        
