# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

from .carter import CARTER_V1_CFG
from .goal import WAYPOINT_CFG

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
    action_space = 2 # Number of actions the neural network should return (wheel velocities)
    # Number of observations fed to the neural network
    observation_space = 6
    state_space = 0

    env_spacing = 10.0 # Spacing between environments, depends on the amount of goals
    num_goals = 1 # Number of goals in the environment

    course_length_coefficient = 2.5 # Coefficient for the length of the course
    course_width_coefficient = 2.0 # Coefficient for the width of the course

    # simulation frames Hz
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CARTER_V1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    wheels = ["left_wheel", "right_wheel"]
    caster = ["rear_pivot", "rear_axle"]

    # goal waypoints
    waypoint_cfg: VisualizationMarkersCfg = WAYPOINT_CFG

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

        self._num_goals = self.cfg.num_goals
        self.env_spacing = self.cfg.env_spacing
        self.course_length_coefficient = self.cfg.course_length_coefficient
        self.course_width_coefficient = self.cfg.course_width_coefficient

        self._wheels_idx, _ = self.carter.find_joints(self.cfg.wheels)
        self._caster_idx, _ = self.carter.find_joints(self.cfg.caster)

        self._goal_reached = torch.zeros((self.num_envs), dtype=torch.int32, device=self.device)
        self.task_completed = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device)

        self._goal_positions = torch.zeros((self.num_envs, self._num_goals,2), dtype=torch.float32, device=self.device)
        self._goal_index = torch.zeros((self.num_envs), dtype=torch.int32, device=self.device)
        self._marker_position = torch.zeros((self.num_envs, self._num_goals, 3), dtype=torch.float32, device=self.device)
        
        # Reward coefficients
        self.linear_velocity_min = 0.0
        self.linear_velocity_max = 1.0
        self.angular_velocity_min = 0.0
        self.angular_velocity_max = 1.0

    def _setup_scene(self):
        self.carter = Articulation(self.cfg.robot_cfg)
        
        # add lidar
        self.lidar = RayCaster(self.cfg.lidar)
        
        # add goal waypoints
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)

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
        # Base velocity scale
        base_velocity_scale = 10.0

        # For differential drive, we directly map neural network outputs to wheel velocities
        # This enables turning by creating a velocity differential between wheels

        # For a 2-dimensional action space (common in RL):
        # actions[:, 0] = left wheel velocity
        # actions[:, 1] = right wheel velocity

        # Apply scaling
        left_wheel_velocity = -actions[:, 0] * base_velocity_scale
        right_wheel_velocity = -actions[:, 1] * base_velocity_scale

        # Create the final action tensor
        self.actions = torch.stack([left_wheel_velocity, right_wheel_velocity], dim=1)

        # Prepare caster action (caster is passive, no control needed)
        self._caster_action = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        # Print some diagnostic information
        if self.num_envs > 0:  # Just print first environment's actions for debugging
            print(f"[DEBUG]Left wheel: {left_wheel_velocity[0].item():.2f}, Right wheel: {right_wheel_velocity[0].item():.2f}")
                
    def _apply_action(self) -> None:

        # Print wheel velocities for the first environment
        if self.num_envs > 0:
            print(f"[DEBUG]Applied wheel velocities - Left: {self.actions[0, 0].item():.2f}, Right: {self.actions[0, 1].item():.2f}")
        self.carter.set_joint_velocity_target(self.actions, joint_ids=self._wheels_idx)
        self.carter.set_joint_position_target(self._caster_action, joint_ids=self._caster_idx)

    def _get_observations(self) -> dict:
        # TODO: Implement observation function

        # Get current goal position
        current_goal_position = self._goal_positions[self.carter._ALL_INDICES, self._goal_index]

        # Calculate position error vector and magnitude
        position_error_vector = current_goal_position - self.carter.data.root_pos_w[:, :2]
        position_error = torch.norm(position_error_vector, dim=-1)

        # Calculate heading error
        heading = self.carter.data.heading_w
        target_heading_w = torch.atan2(
            current_goal_position[:, 1] - self.carter.data.root_pos_w[:, 1],
            current_goal_position[:, 0] - self.carter.data.root_pos_w[:, 0]
        )
        heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        # Combine observations
        obs = torch.cat(
            (
                position_error.unsqueeze(dim=1),
                torch.cos(heading_error).unsqueeze(dim=1),
                torch.sin(heading_error).unsqueeze(dim=1),
                self.carter.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),  # Forward velocity
                self.carter.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),  # Lateral velocity
                self.carter.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),  # Angular velocity (yaw)
            ),
            dim=-1
        )

        # Update the observation space to match
        self.cfg.observation_space = obs.shape[1]

        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # TODO: Implement reward function
        
        # Constants for reward calculation
        position_tolerance = 0.3  # Increased tolerance for easier goal detection
        goal_reached_bonus = 20.0  # Bigger bonus for reaching a goal
        heading_alignment_weight = 0.2  # Weight for aligning heading with goal
        distance_progress_weight = 0.5  # Weight for getting closer to goal

        # Get current goal position
        current_goal_position = self._goal_positions[self.carter._ALL_INDICES, self._goal_index]

        # Calculate position error
        position_error_vector = current_goal_position - self.carter.data.root_pos_w[:, :2]
        position_error = torch.norm(position_error_vector, dim=-1)

        # Store previous position error for progress calculation if it doesn't exist
        if not hasattr(self, '_previous_position_error'):
            self._previous_position_error = position_error.clone()

        # Calculate distance progress (positive when getting closer to goal)
        distance_progress = self._previous_position_error - position_error

        # Update previous position error for next step
        self._previous_position_error = position_error.clone()

        # Check if goal is reached
        goal_reached = position_error < position_tolerance

        # Calculate heading error and alignment reward
        heading = self.carter.data.heading_w
        target_heading_w = torch.atan2(
            position_error_vector[:, 1],
            position_error_vector[:, 0]
        )
        heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        heading_alignment = torch.cos(heading_error)  # 1 when perfectly aligned, -1 when facing opposite

        # Calculate composite reward
        composite_reward = (
            heading_alignment * heading_alignment_weight +
            distance_progress * distance_progress_weight +
            goal_reached * goal_reached_bonus
        )

        # Update goal index if goal reached
        self._goal_index = torch.where(goal_reached, self._goal_index + 1, self._goal_index)
        self.task_completed = self._goal_index >= self._num_goals
        self._goal_index = self._goal_index % self._num_goals

        # Update waypoint visualization
        one_hot_encoded = torch.nn.functional.one_hot(self._goal_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

        return composite_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Task fails if episode length is exceeded
        failure_termination = self.episode_length_buf >= self.max_episode_length - 1

        # Task completes successfully if all goals are reached
        # This is determined in _get_rewards and stored in self.task_completed
        clean_termination = self.task_completed

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

        # Reset goal
        self._goal_positions[env_ids, :, :] = 0.0
        self._marker_position[env_ids, :, :] = 0.0

        spacing = 2 / self._num_goals
        goal_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        self._goal_positions[env_ids, :len(goal_positions), 0] = goal_positions
        self._goal_positions[env_ids, :, 1] = torch.rand((num_reset, self._num_goals), device=self.device, dtype=torch.float32) + self.course_length_coefficient
        self._goal_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        self._goal_index[env_ids] = 0

        # Reset markers
        self._marker_position[env_ids, :, :2] = self._goal_positions[env_ids]
        visualise_pos = self._marker_position.view(-1, 3)
        self.waypoints.visualize(translations=visualise_pos)
