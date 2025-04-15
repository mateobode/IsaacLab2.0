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
    decimation = 4 # Decimation factor for rendering
    episode_length_s = 20.0 # Maximum episode length in seconds
    action_space = 2 # Number of actions the neural network should return (wheel velocities)
    # Number of observations fed to the neural network
    observation_space = 6
    state_space = 0

    env_spacing = 30.0 # Spacing between environments, depends on the amount of goals
    num_goals = 10 # Number of goals in the environment

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
        
        # Reward parameters
        self.position_tolerance: float = 0.15 # Tolerance for the position of the robot
        self.goal_reached_reward = 10.0
        self.position_progress_weight = 1.0
        self.heading_progress_weight = 0.05

    def _setup_scene(self):
        self.carter = Articulation(self.cfg.robot_cfg)
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False) # Clone child environments from parent environment
        self.scene.filter_collisions(global_prim_paths=[]) # Prevent environments from colliding with each other

        # add articulation to scene
        self.scene.articulations["carter"] = self.carter

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        base_velocity_scale = 10.0

        # For differential drive, we directly map neural network outputs to wheel velocities
        # This enables turning by creating a velocity differential between wheels

        # For a 2-dimensional action space (common in RL):
        # actions[:, 0] = left wheel velocity
        # actions[:, 1] = right wheel velocity

        # Scaling
        left_wheel_velocity = -actions[:, 0] * base_velocity_scale
        right_wheel_velocity = -actions[:, 1] * base_velocity_scale

        # Create the final action tensor
        self.actions = torch.stack([left_wheel_velocity, right_wheel_velocity], dim=1)

        # Passive caster
        self._caster_action = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
                
    def _apply_action(self) -> None:
        self.carter.set_joint_velocity_target(self.actions, joint_ids=self._wheels_idx)
        self.carter.set_joint_position_target(self._caster_action, joint_ids=self._caster_idx)

    def _get_observations(self) -> dict:
        # Calculate position error
        current_goal_position = self._goal_positions[self.carter._ALL_INDICES, self._goal_index]
        self._position_error_vector = current_goal_position - self.carter.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        # Calculate heading error
        heading = self.carter.data.heading_w
        target_heading_w = torch.atan2(
            self._goal_positions[self.carter._ALL_INDICES, self._goal_index, 1] - self.carter.data.root_link_pos_w[:, 1],
            self._goal_positions[self.carter._ALL_INDICES, self._goal_index, 0] - self.carter.data.root_link_pos_w[:, 0],
        )
        self.goal_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        # Combine observations
        obs = torch.cat(
            (
                self._position_error.unsqueeze(dim=1),
                torch.cos(self.goal_heading_error).unsqueeze(dim=1),
                torch.sin(self.goal_heading_error).unsqueeze(dim=1),
                self.carter.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),  # Forward velocity
                self.carter.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),  # Lateral velocity
                self.carter.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),  # Angular velocity (yaw)
            ),
            dim=-1
        )
        
        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Jit TorchScript for calculating rewards
        composite_reward, self.task_completed, self._goal_index = compute_rewards(
            self.position_tolerance,
            self._num_goals,
            self.position_progress_weight,
            self.heading_progress_weight,
            self.goal_reached_reward,
            self._previous_position_error,
            self._position_error,
            self.goal_heading_error,
            self.task_completed,
            self._goal_index,
        )

        # Update waypoint visualization
        one_hot_encoded = torch.nn.functional.one_hot(self._goal_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)
        
        return composite_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = self.episode_length_buf > self.max_episode_length

        # Task completed is determined in _get_rewards
        return task_failed, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.carter._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # Reset robot
        default_state = self.carter.data.default_root_state[env_ids] # First there are pos, next 4 quats, next 3 vel, next 3 ang vel
        carter_pose = default_state[:, :7]  # Get default pose from config file
        carter_velocities = default_state[:, 7:] # Get default velocities from config file
        joint_positions = self.carter.data.default_joint_pos[env_ids] # Get joint positions from config file
        joint_velocities = self.carter.data.default_joint_vel[env_ids] # Get joint velocities from config file

        carter_pose[:, :3] += self.scene.env_origins[env_ids]  # Adds center of each environment position in carter position

        # Randomize starting position (not too far from origin)
        carter_pose[:, 0] -= self.env_spacing / 2  # Center the robot
        carter_pose[:, 1] += 2.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device) * self.course_width_coefficient

        # Randomize starting heading
        angles = torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device)

        # Isaac Sim Quaternions are w first (w, x, y, z)
        # To rotate about the Z axis, we will modify the W and Z values
        carter_pose[:, 3] = torch.cos(angles * 0.5)  # W quaternion
        carter_pose[:, 6] = torch.sin(angles * 0.5)  # Z quaternion

        # Write to simulation
        self.carter.write_root_pose_to_sim(carter_pose, env_ids)
        self.carter.write_root_velocity_to_sim(carter_velocities, env_ids)
        self.carter.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)

        # Reset goals
        self._goal_positions[env_ids, :, :] = 0.0
        self._marker_position[env_ids, :, :] = 0.0

        spacing = 2 / self._num_goals
        goal_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        self._goal_positions[env_ids, :len(goal_positions), 0] = goal_positions
        self._goal_positions[env_ids, :, 1] = torch.rand((num_reset, self._num_goals), dtype=torch.float32, device=self.device) * self.course_length_coefficient
        self._goal_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        # Reset goal index
        self._goal_index[env_ids] = 0

        # Reset visual markers
        self._marker_position[env_ids, :, :2] = self._goal_positions[env_ids]
        visualize_pos = self._marker_position.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)

        # Reset position error
        current_goal_position = self._goal_positions[self.carter._ALL_INDICES, self._goal_index]
        self._position_error_vector = current_goal_position[:, :2] - self.carter.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        # Reset heading error
        heading = self.carter.data.heading_w[:]
        target_heading_w = torch.atan2(
            self._goal_positions[:, 0, 1] - self.carter.data.root_link_pos_w[:, 1],
            self._goal_positions[:, 0, 0] - self.carter.data.root_link_pos_w[:, 0],
        )
        self.goal_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self._previous_heading_error = self.goal_heading_error.clone()

@torch.jit.script
def compute_rewards(
    position_tolerance: float,
    _num_goals: int,
    position_progress_weight: float,
    heading_progress_weight: float,
    goal_reached_reward: float,
    _previous_position_error: torch.Tensor,
    _position_error: torch.Tensor,
    goal_heading_error: torch.Tensor,
    task_completed: torch.Tensor,
    _goal_index: torch.Tensor, 
):
    # Position progress
    position_progress_reward = _previous_position_error - _position_error
    # Heading progress (1 when perfectly aligned, -1 when facing opposite)
    goal_heading_reward = torch.cos(goal_heading_error)
    # Check if goal is reached
    goal_reached = _position_error < position_tolerance
    # If goal is reached, the goal index is updated
    _goal_index = _goal_index + goal_reached
    task_completed = _goal_index > (_num_goals-1)
    _goal_index = _goal_index % _num_goals

    composite_reward = (
        position_progress_reward*position_progress_weight +
        goal_heading_reward*heading_progress_weight +
        goal_reached*goal_reached_reward
    )

    if torch.any(composite_reward.isnan()):
        raise ValueError("Rewards cannot be NAN")
    
    return composite_reward, task_completed, _goal_index
