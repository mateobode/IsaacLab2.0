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
from .walls import WALL_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
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
    num_walls = 9 # Number of walls in the environment

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

    # walls
    wall_collection_cfg: RigidObjectCollectionCfg = WALL_CFG

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

class CarterEnv(DirectRLEnv):
    cfg: CarterEnvCfg

    def __init__(self, cfg: CarterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._num_goals = self.cfg.num_goals
        self._num_walls = self.cfg.num_walls
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
        
        # add goal waypoints
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)

        # add walls
        self.walls = RigidObjectCollection(self.cfg.wall_collection_cfg)

        # add ground plane
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
        #if self.num_envs > 0:  # Just print first environment's actions for debugging
        #    print(f"[DEBUG]Left wheel: {left_wheel_velocity[0].item():.2f}, Right wheel: {right_wheel_velocity[0].item():.2f}")
                
    def _apply_action(self) -> None:

        # Print wheel velocities for the first environment
        #if self.num_envs > 0:
        #    print(f"[DEBUG]Applied wheel velocities - Left: {self.actions[0, 0].item():.2f}, Right: {self.actions[0, 1].item():.2f}")
        self.carter.set_joint_velocity_target(self.actions, joint_ids=self._wheels_idx)
        self.carter.set_joint_position_target(self._caster_action, joint_ids=self._caster_idx)

    def _get_observations(self) -> dict:
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
        position_tolerance = 0.3
        goal_reached_bonus = 20.0
        position_progress_weight = 1.0
        heading_alignment_weight = 0.2
        
        # Get current goal position
        current_goal_position = self._goal_positions[self.carter._ALL_INDICES, self._goal_index]
        
        # Calculate position error
        position_error_vector = current_goal_position - self.carter.data.root_pos_w[:, :2]
        position_error = torch.norm(position_error_vector, dim=-1)
        
        # Store previous position error for next time if it doesn't exist
        if not hasattr(self, '_previous_position_error'):
            self._previous_position_error = position_error.clone()
        
        # Calculate heading error
        heading = self.carter.data.heading_w
        target_heading_w = torch.atan2(
            position_error_vector[:, 1],
            position_error_vector[:, 0]
        )
        heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        
        # Use JIT-compiled function for reward calculation
        composite_reward, self.task_completed, self._goal_index = compute_rewards(
            position_tolerance,
            goal_reached_bonus,
            position_progress_weight,
            heading_alignment_weight,
            self._num_goals,
            self._previous_position_error,
            position_error,
            heading_error,
            self._goal_index,
            self.task_completed
        )
        
        # Update previous position error for next step
        self._previous_position_error = position_error.clone()
        
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
        carter_pose = default_state[:, :7].clone()  # Make sure to clone
        carter_velocities = default_state[:, 7:].clone()
        joint_positions = self.carter.data.default_joint_pos[env_ids].clone()
        joint_velocities = self.carter.data.default_joint_vel[env_ids].clone()

        # Add env origins
        carter_pose[:, :3] += self.scene.env_origins[env_ids]  # Note: ADD to current positions

        # Randomize starting position (but don't move too far from origin)
        carter_pose[:, 0] -= self.env_spacing / 2  # Center the robot
        carter_pose[:, 1] += 2.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device) * self.course_width_coefficient

        # Randomize starting heading
        angles = torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device)

        # Set quaternion for Z-axis rotation (yaw)
        carter_pose[:, 3] = torch.cos(angles * 0.5)  # w component
        carter_pose[:, 6] = torch.sin(angles * 0.5)  # z component

        # Write to simulation
        self.carter.write_root_pose_to_sim(carter_pose, env_ids)
        self.carter.write_root_velocity_to_sim(carter_velocities, env_ids)
        self.carter.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)

        # Reset goals
        self._goal_positions[env_ids, :, :] = 0.0
        self._marker_position[env_ids, :, :] = 0.0

        # Wall spacing constraints
        wall_half_width = 0.05
        course_width = self.course_width_coefficient

        # Calculate safe area between walls
        safe_margin = 0.2
        min_y_local = -course_width/2 + wall_half_width + safe_margin
        max_y_local = course_width/2 - wall_half_width - safe_margin

        # Set up goal positions
        spacing = 2 / self._num_goals
        goal_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        self._goal_positions[env_ids, :len(goal_positions), 0] = goal_positions

        # Generate Y positions of goals between walls
        y_positions = min_y_local + torch.rand((len(env_ids), self._num_goals), dtype=torch.float32, device=self.device) * (max_y_local - min_y_local)
        self._goal_positions[env_ids, :, 1] = y_positions

        # Add environment origins to goal positions
        self._goal_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        # For each environment
        for env_idx, env_id in enumerate(env_ids):
            # Get waypoint positions for this environment
            waypoints = self._goal_positions[env_id, :, :]

            # Number of segments to position (limited by min of waypoints-1 and wall segments)
            num_segments = min(self._num_walls, self._num_goals - 1)

            # For each wall segment
            for segment in range(num_segments):
                # Get waypoint positions that this segment connects
                start_idx = segment
                end_idx = segment + 1
                #if start_idx >= len(waypoints) or end_idx >= len(waypoints):
                #    continue

                # Get waypoint positions (in world coordinates)
                start_pos = waypoints[start_idx, :2]
                end_pos = waypoints[end_idx, :2]

                # Calculate segment direction
                segment_direction = end_pos - start_pos
                segment_length = torch.norm(segment_direction)

                if segment_length > 0:  # Avoid division by zero
                    # Normalize direction
                    segment_direction = segment_direction / segment_length

                    # Calculate perpendicular direction for wall placement
                    perp_direction = torch.tensor([-segment_direction[1], segment_direction[0]], device=self.device)

                    # Position of left and right walls (at segment midpoint)
                    segment_midpoint = (start_pos + end_pos) / 2
                    corridor_width = self.course_width_coefficient

                    # Left wall position
                    left_wall_pos = segment_midpoint + perp_direction * (-corridor_width/2)
                    # Right wall position
                    right_wall_pos = segment_midpoint + perp_direction * (corridor_width/2)

                    # Wall orientation (rotation around Z-axis to align with segment)
                    wall_angle = torch.atan2(segment_direction[1], segment_direction[0])

                    # Convert to quaternion (rotation around Z-axis)
                    wall_quat_w = torch.cos(wall_angle * 0.5)
                    wall_quat_z = torch.sin(wall_angle * 0.5)
                    wall_quat = torch.tensor([wall_quat_w, 0.0, 0.0, wall_quat_z], device=self.device)

                    # Create position tensors
                    left_wall_pos_tensor = torch.zeros(3, device=self.device)
                    left_wall_pos_tensor[0] = left_wall_pos[0]
                    left_wall_pos_tensor[1] = left_wall_pos[1]
                    left_wall_pos_tensor[2] = 0.25  # Half height above ground

                    right_wall_pos_tensor = torch.zeros(3, device=self.device)
                    right_wall_pos_tensor[0] = right_wall_pos[0]
                    right_wall_pos_tensor[1] = right_wall_pos[1]
                    right_wall_pos_tensor[2] = 0.25  # Half height above ground

                    # Create pose tensors (concatenate position and quaternion)
                    left_wall_pose = torch.cat([left_wall_pos_tensor, wall_quat])
                    right_wall_pose = torch.cat([right_wall_pos_tensor, wall_quat])

                    # Calculate object indices for left and right walls
                    left_wall_obj_id = torch.tensor([segment], device=self.device)
                    right_wall_obj_id = torch.tensor([segment + self._num_walls], device=self.device)

                    # Single environment ID tensor
                    env_id_tensor = torch.tensor([env_idx], device=self.device)

                    # Write wall poses to simulation
                    try:
                        # Set left wall pose
                        self.walls.write_object_pose_to_sim(
                            left_wall_pose.unsqueeze(0).unsqueeze(0),  # Shape: [1, 1, 7]
                            env_ids=env_id_tensor,
                            object_ids=left_wall_obj_id
                        )

                        # Set right wall pose
                        self.walls.write_object_pose_to_sim(
                            right_wall_pose.unsqueeze(0).unsqueeze(0),  # Shape: [1, 1, 7]
                            env_ids=env_id_tensor,
                            object_ids=right_wall_obj_id
                        )
                    except Exception as e:
                        print(f"Error setting wall position: {e}")

        # Reset goal index
        self._goal_index[env_ids] = 0

        # Update position error
        current_goal_position = self._goal_positions[self.carter._ALL_INDICES, self._goal_index]
        position_error_vector = current_goal_position - self.carter.data.root_pos_w[:, :2]
        position_error = torch.norm(position_error_vector, dim=-1)
        if hasattr(self, '_previous_position_error'):
            self._previous_position_error[env_ids] = position_error[env_ids].clone()
        else:
            self._previous_position_error = position_error.clone()

        # Reset markers
        self._marker_position[env_ids, :, :2] = self._goal_positions[env_ids]
        visualize_pos = self._marker_position.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)



@torch.jit.script
def compute_rewards(
    position_tolerance: float,
    goal_reached_bonus: float,
    position_progress_weight: float,
    heading_alignment_weight: float,
    num_goals: int,
    previous_position_error: torch.Tensor,
    position_error: torch.Tensor,
    heading_error: torch.Tensor,
    goal_index: torch.Tensor,
    task_completed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Calculate distance progress (positive when getting closer to goal)
    position_progress = previous_position_error - position_error
    
    # Check if goal is reached
    goal_reached = position_error < position_tolerance
    
    # Calculate heading alignment (1 when perfectly aligned, -1 when facing opposite)
    heading_alignment = torch.cos(heading_error)
    
    # Calculate composite reward
    composite_reward = (
        position_progress * position_progress_weight +
        heading_alignment * heading_alignment_weight +
        goal_reached * goal_reached_bonus
    )
    
    # Increment goal index for environments that reached their goal
    new_goal_index = torch.where(goal_reached, goal_index + 1, goal_index)
    
    # Check if task is completed (all goals reached)
    new_task_completed = new_goal_index >= num_goals
    
    # Wrap around the goal index
    new_goal_index = new_goal_index % num_goals
    
    # Check for NaN values in rewards
    if torch.any(composite_reward.isnan()):
        raise ValueError("Rewards cannot be NAN")
    
    return composite_reward, new_task_completed, new_goal_index
