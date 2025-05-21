# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import omni.usd

from collections.abc import Sequence
from typing import List, Optional, Union, cast

from .carter import CARTER_V1_CFG
from .goal import WAYPOINT_CFG
from .walls import WALL_CFG, WALLS_CFG


import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCollection, RigidObjectCollectionCfg

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# Add imports for contact sensors
from isaaclab.sensors import ContactSensor, ContactSensorCfg
# Import the physx interface for LiDAR
from isaacsim.sensors.physx import _range_sensor


@configclass
class CarterEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4 # Decimation factor for rendering
    episode_length_s = 20.0 # Maximum episode length in seconds
    action_space = 2 # Number of actions the neural network should return (wheel velocities)
    # Number of observations fed to the neural network
    observation_space = 38 # Updated: base observations (6) + LiDAR (32)
    state_space = 0

    env_spacing = 30.0 # Spacing between environments, depends on the amount of goals
    num_goals = 10 # Number of goals in the environment
    lidar_num_sectors = 32  # Divide 360° LIDAR into this many sectors
    lidar_max_range = 100.0  # Maximum detection range

    course_length_coefficient = 2.5 # Coefficient for the length of the course
    course_width_coefficient = 2.0 # Coefficient for the width of the course

    # simulation frames Hz
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CARTER_V1_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore[attr-defined]
    wheels = ["left_wheel", "right_wheel"]

    # goal waypoints
    waypoint_cfg: VisualizationMarkersCfg = WAYPOINT_CFG

    # walls
    wall_collection_cfg: RigidObjectCollectionCfg = WALL_CFG

    # Contact sensor configuration
    contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/chassis_link",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["/World/{ENV_REGEX_NS}/Wall_.*"],  # Walls are our obstacles
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
        self._lidar_num_sectors = self.cfg.lidar_num_sectors
        self._lidar_max_range = self.cfg.lidar_max_range

        self._wheels_idx, _ = self.carter.find_joints(self.cfg.wheels)

        self._goal_reached = torch.zeros((self.num_envs), dtype=torch.int32, device=self.device)
        self.task_completed = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device)

        self._goal_positions = torch.zeros((self.num_envs, self._num_goals,2), dtype=torch.float32, device=self.device)
        self._goal_index = torch.zeros((self.num_envs), dtype=torch.int32, device=self.device)
        self._marker_position = torch.zeros((self.num_envs, self._num_goals, 3), dtype=torch.float32, device=self.device)

        # New collision penalty parameters
        self.collision_penalty: float = 30.0  # Large penalty for collisions

        self.proximity_penalty_scale: float = 5.0  # Penalty scale for proximity to obstacles
        # Add minimum distance threshold - no penalty beyond this distance
        self.proximity_min_distance: float = 0.8  # Only apply penalty when closer than this distance (meters)

        # New collision detection flag - ensure it has the right shape from the beginning
        # Make sure it's initialized with shape [num_envs]
        self.collision_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # LiDAR interface for processing data
        self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
        # Store LiDAR paths for each environment
        self.lidar_paths = []

        # Reward parameters
        self.position_tolerance: float = 0.5 # Tolerance for the position of the robot
        self.goal_reached_reward: float = 50.0
        self.position_progress_weight: float = 1.0
        self.heading_progress_weight: float = 1.0

        # Debug flag - set to True to print penalty and distance information
        self.debug_penalties = False

    def _find_lidar_paths(self):
        """Find all LiDAR paths in the scene for each environment."""
        stage = omni.usd.get_context().get_stage()
        self.lidar_paths = []
        
        # For each environment, find the LiDAR path
        for env_idx in range(self.num_envs):
            # Try with environment namespace
            test_path = f"/World/envs/env_{env_idx}/Robot/chassis_link/carter_lidar"
            if stage.GetPrimAtPath(test_path).IsValid():
                self.lidar_paths.append(test_path)
            else:
                # If not found, try with just the environment number
                test_path = f"/World/env_{env_idx}/Robot/chassis_link/carter_lidar"
                if stage.GetPrimAtPath(test_path).IsValid():
                    self.lidar_paths.append(test_path)
                else:
                    # If still not found, log a warning
                    print(f"Warning: Could not find LiDAR for environment {env_idx}")
                    self.lidar_paths.append(None)
                    
        # If we couldn't find any LiDAR paths, try a more general search
        if all(path is None for path in self.lidar_paths):
            print("Searching for any LiDAR in the scene...")
            for prim in stage.Traverse():
                if "lidar" in prim.GetPath().pathString.lower():
                    print(f"Found potential LiDAR at: {prim.GetPath()}")
                    # If we find one, use it for all environments
                    self.lidar_paths = [str(prim.GetPath())] * self.num_envs
                    break

    def _setup_scene(self):
        self.carter = Articulation(self.cfg.robot_cfg)
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        self.walls = RigidObjectCollection(self.cfg.wall_collection_cfg)
        self.wall_state = []
        
        # Initialize the contact sensor
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False) # Clone child environments from parent environment
        self.scene.filter_collisions(global_prim_paths=[]) # Prevent environments from colliding with each other
        # add articulation to scene
        self.scene.articulations["carter"] = self.carter
        # add walls as collection to scene
        self.scene.rigid_object_collections["walls"] = self.walls
        # add sensors to scene
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        # Find all LiDAR paths after scene setup
        self._find_lidar_paths()

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        action_scale = 10.0
        self.actions = actions.clone() * action_scale

    def _apply_action(self) -> None:
        self.carter.set_joint_velocity_target(self.actions, joint_ids=self._wheels_idx)

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

        # Process LiDAR data using the PhysX interface
        lidar_sectors = torch.ones((self.num_envs, self._lidar_num_sectors), device=self.device) * self._lidar_max_range
        
        for env_idx in range(self.num_envs):
            if env_idx < len(self.lidar_paths) and self.lidar_paths[env_idx] is not None:
                try:
                    # Get depth data from the LiDAR
                    depth_data = self.lidar_interface.get_linear_depth_data(self.lidar_paths[env_idx])
                    if depth_data is not None and len(depth_data) > 0:
                        # Convert numpy array to torch tensor
                        depth_tensor = torch.tensor(depth_data, device=self.device)
                        
                        # Reshape if needed based on the data format
                        if len(depth_tensor.shape) > 1:
                            # If 2D, flatten to 1D
                            depth_tensor = depth_tensor.flatten()
                        
                        # Bin the data into sectors
                        num_points = len(depth_tensor)
                        points_per_sector = max(1, num_points // self._lidar_num_sectors)
                        
                        for i in range(self._lidar_num_sectors):
                            start_idx = i * points_per_sector
                            end_idx = min((i + 1) * points_per_sector, num_points)
                            
                            if start_idx < end_idx:
                                # Get minimum distance in each sector
                                sector_data = depth_tensor[start_idx:end_idx]
                                # Replace infinity values with max range
                                sector_data = torch.where(
                                    torch.isinf(sector_data),
                                    torch.tensor(self._lidar_max_range, device=self.device),
                                    sector_data
                                )
                                if len(sector_data) > 0:
                                    lidar_sectors[env_idx, i] = torch.min(sector_data)
                except Exception as e:
                    print(f"Error reading LiDAR data for env {env_idx}: {e}")
        
        # Normalize LiDAR readings to [0, 1] range
        lidar_sectors = lidar_sectors / self._lidar_max_range
        
        # Check for collisions using contact sensor
        contact_forces = self.contact_sensor.data.net_forces_w
        
        # For debugging:
        # print(f"contact_forces shape: {contact_forces.shape}, num_envs: {self.num_envs}")
        
        # Handle the contact forces tensor properly based on its shape
        if len(contact_forces.shape) > 2:
            # If we have a 3D tensor, flatten all but the first dimension
            reshaped_forces = contact_forces.reshape(contact_forces.shape[0], -1)
            contact_magnitude = torch.norm(reshaped_forces, dim=1)
        else:
            # Standard case: calculate norm along the last dimension
            contact_magnitude = torch.norm(contact_forces, dim=-1)
        
        # If contact_magnitude has more elements than num_envs, 
        # take the maximum value per environment
        if contact_magnitude.shape[0] > self.num_envs:
            # Reshape to [num_envs, N]
            n_per_env = contact_magnitude.shape[0] // self.num_envs
            contact_magnitude = contact_magnitude.reshape(self.num_envs, n_per_env)
            # Take max for each environment
            contact_magnitude = torch.max(contact_magnitude, dim=1)[0]
        
        # Now contact_magnitude should be [num_envs]
        self.collision_detected = contact_magnitude > 0.1  # Force threshold for collision detection

        # Combine observations
        base_obs = torch.cat(
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
        
        # Combine base observations with LiDAR data
        obs = torch.cat((base_obs, lidar_sectors), dim=1)
        
        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Original reward calculation from the jit TorchScript
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

        # Add proximity penalty from LiDAR data
        proximity_penalty = torch.zeros((self.num_envs), device=self.device)
        
        for env_idx in range(self.num_envs):
            if env_idx < len(self.lidar_paths) and self.lidar_paths[env_idx] is not None:
                try:
                    depth_data = self.lidar_interface.get_linear_depth_data(self.lidar_paths[env_idx])
                    if depth_data is not None and len(depth_data) > 0:
                        # Convert to tensor and handle infinite values
                        depth_tensor = torch.tensor(depth_data, device=self.device)
                        depth_tensor = torch.where(
                            torch.isinf(depth_tensor),
                            torch.tensor(self._lidar_max_range, device=self.device),
                            depth_tensor
                        )
                        
                        # Calculate minimum distance to obstacles
                        min_distance = torch.min(depth_tensor)
                        
                        # Apply proximity penalty - only if closer than min distance threshold
                        if min_distance < self.proximity_min_distance:
                            # Calculate how close we are to the minimum safe distance (0 to 1 scale)
                            # 0 means at the min distance, 1 means touching the obstacle
                            proximity_ratio = 1.0 - (min_distance / self.proximity_min_distance)
                            
                            # Apply a quadratic penalty curve that increases more sharply as the robot gets very close
                            # This gives a more gradual penalty at medium distances but still strong avoidance when very close
                            proximity_penalty[env_idx] = self.proximity_penalty_scale * (proximity_ratio ** 2)
                            
                            # Debug information
                            if self.debug_penalties and env_idx == 0:  # Only print for the first environment to avoid spam
                                print(f"Min distance: {min_distance:.2f}m, Proximity ratio: {proximity_ratio:.2f}, Penalty: {proximity_penalty[env_idx]:.2f}")
                except Exception as e:
                    print(f"Error calculating proximity penalty for env {env_idx}: {e}")
        
        # Apply collision penalty
        collision_penalty = self.collision_detected * self.collision_penalty
        
        # Debug information for collision detection
        if self.debug_penalties and torch.any(self.collision_detected):
            num_collisions = torch.sum(self.collision_detected).item()
            print(f"Collisions detected: {num_collisions}, Penalty per collision: {self.collision_penalty}")
        
        # Add penalties to the original reward
        original_reward = composite_reward.clone()
        composite_reward = composite_reward - proximity_penalty - collision_penalty
        
        # Debug reward components
        if self.debug_penalties:
            # Calculate average penalties across environments
            avg_proximity_penalty = torch.mean(proximity_penalty).item()
            avg_collision_penalty = torch.mean(collision_penalty).item()
            avg_original_reward = torch.mean(original_reward).item()
            avg_final_reward = torch.mean(composite_reward).item()
            
            # Only print every 20 steps to reduce spam
            if self.episode_length_buf[0] % 20 == 0:
                print("\nReward components:")
                print(f"Original reward: {avg_original_reward:.2f}")
                print(f"Proximity penalty: {avg_proximity_penalty:.2f}")
                print(f"Collision penalty: {avg_collision_penalty:.2f}")
                print(f"Final reward: {avg_final_reward:.2f}")
                print("-------------------")

        # Update waypoint visualization
        one_hot_encoded = torch.nn.functional.one_hot(self._goal_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)
        
        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        return composite_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Debug information
        # print(f"episode_length_buf shape: {self.episode_length_buf.shape}, collision_detected shape: {self.collision_detected.shape}")
        
        # Make sure both tensors have the same shape before logical_or
        episode_timeout = self.episode_length_buf > self.max_episode_length
        
        # Ensure the collision_detected tensor has the right shape
        if self.collision_detected.shape != episode_timeout.shape:
            try:
                # Try reshaping if possible (should be broadcast-compatible)
                if self.collision_detected.numel() == self.num_envs:
                    self.collision_detected = self.collision_detected.reshape(episode_timeout.shape)
                elif self.collision_detected.numel() > self.num_envs:
                    # If we have more elements, take the ones we need
                    self.collision_detected = self.collision_detected[:self.num_envs]
                else:
                    # If we have too few elements, pad with False
                    print(f"Warning: collision_detected has too few elements: {self.collision_detected.shape}")
                    padded = torch.zeros(episode_timeout.shape, dtype=torch.bool, device=self.device)
                    padded[:self.collision_detected.numel()] = self.collision_detected
                    self.collision_detected = padded
            except Exception as e:
                # In case of errors, just ignore collisions for safety
                print(f"Error reshaping collision_detected: {e}")
                print(f"episode_timeout: {episode_timeout.shape}, collision_detected: {self.collision_detected.shape}")
                self.collision_detected = torch.zeros_like(episode_timeout)
        
        # Task failed conditions with more explicit error handling
        try:
            task_failed = torch.logical_or(episode_timeout, self.collision_detected)
        except Exception as e:
            print(f"Error in logical_or: {e}")
            # Fallback to just using episode_timeout
            task_failed = episode_timeout
        
        # Task completed is determined in _get_rewards
        return task_failed, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Keep the original implementation without modifications
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

        # Reset walls
        num_walls = len(WALLS_CFG.rigid_objects)
        self.wall_state = self.walls.data.default_object_state.clone()
        wall_ids = torch.arange(num_walls, device=self.device)
    
        # Simple corridor parameters
        corridor_width = 4.0  # Wider corridor for easier navigation
        wall_z_height = 0.25  # Standard height for walls
        
        for i in range(num_reset):
            # We'll create a simple corridor along the path
            for j in range(self._num_goals):
                goal_pos = self._goal_positions[env_ids[i], j]
                
                # For each goal, place 2 walls (one on each side)
                if 2*j < num_walls:
                    # Left wall - offset to the left of the goal
                    self.wall_state[env_ids[i], 2*j, 0] = goal_pos[0]  # Same x as goal
                    self.wall_state[env_ids[i], 2*j, 1] = goal_pos[1] + corridor_width/2  # Offset in y
                    self.wall_state[env_ids[i], 2*j, 2] = wall_z_height  # Z position
                    
                    # Right wall - offset to the right of the goal
                    self.wall_state[env_ids[i], 2*j+1, 0] = goal_pos[0]  # Same x as goal
                    self.wall_state[env_ids[i], 2*j+1, 1] = goal_pos[1] - corridor_width/2  # Offset in y
                    self.wall_state[env_ids[i], 2*j+1, 2] = wall_z_height  # Z position
                    
                    # Standard orientation (no rotation) for stability
                    # Quaternion components (w,x,y,z) for identity rotation
                    self.wall_state[env_ids[i], 2*j, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                    self.wall_state[env_ids[i], 2*j+1, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        
        # Set the wall pose in simulation
        self.walls.write_object_link_pose_to_sim(self.wall_state[env_ids, :, :7], env_ids, wall_ids)

        # Clear LiDAR cache and reset sensors for all environments being reset
        for idx, env_id in enumerate(env_ids):
            if env_id < len(self.lidar_paths) and self.lidar_paths[env_id] is not None:
                try:
                    _range_sensor.clear_cache(self.lidar_paths[env_id])
                    self.lidar_interface.reset_sensor(self.lidar_paths[env_id])
                except Exception as e:
                    print(f"Error resetting LiDAR for env {env_id}: {e}")

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
        
        # Reset collision flag - ensure correct shape handling
        try:
            if isinstance(env_ids, torch.Tensor):
                self.collision_detected[env_ids] = False
            else:
                # If env_ids is a list-like object, convert to proper indexing format
                env_ids_tensor = torch.tensor(list(env_ids), dtype=torch.int64, device=self.device) 
                self.collision_detected[env_ids_tensor] = False
        except Exception as e:
            # Fallback: reset all collision flags
            print(f"Warning: Error resetting collision flags: {e}")
            self.collision_detected.fill_(False)

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
