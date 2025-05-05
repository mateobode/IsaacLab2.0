# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import omni
import asyncio
from collections.abc import Sequence

from .carter import CARTER_V1_CFG
from .goal import WAYPOINT_CFG
from .walls import WALL_CFG, WALLS_CFG


import isaaclab.sim as sim_utils
from isaacsim.sensors.physx import _range_sensor
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
    observation_space = 38 # Number of observations fed to the neural network
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
    robot_cfg: ArticulationCfg = CARTER_V1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    wheels = ["left_wheel", "right_wheel"]

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

        self._sector_distances = torch.ones((self.num_envs, self._lidar_num_sectors), dtype=torch.float32, device=self.device) * self._lidar_max_range
        
        # Penalty parameters
        self.obstacle_penalty_weight: float = 0.05
        self.min_safe_distance: float = 0.8
        self.max_lidar_penalty_range: float = 1.5

        # Reward parameters
        self.position_tolerance: float = 0.5 # Tolerance for the position of the robot
        self.goal_reached_reward: float = 20.0
        self.position_progress_weight: float = 1.0
        self.heading_progress_weight: float = 1.0

    def _setup_scene(self):
        self.carter = Articulation(self.cfg.robot_cfg)
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        self.walls = RigidObjectCollection(self.cfg.wall_collection_cfg)
        self.wall_state = []
        
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False) # Clone child environments from parent environment
        self.scene.filter_collisions(global_prim_paths=[]) # Prevent environments from colliding with each other
        # add articulation to scene
        self.scene.articulations["carter"] = self.carter
        # add walls as collection to scene
        self.scene.rigid_object_collections["walls"] = self.walls
        
        # add lidar
        stage = omni.usd.get_context().get_stage()
        self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
        omni.kit.commands.execute('AddPhysicsSceneCommand', stage=stage, path='/World/PhysicsScene')

        for env_idx in range(self.num_envs):
            # Get lidar path for each environment
            env_lidar_path = f"/World/envs/env_{env_idx}/Robot/chassis_link/carter_lidar"
            if self.lidar_interface.is_lidar_sensor(env_lidar_path):
                print("Lidar sensor is valid!")
            result, _ = omni.kit.commands.execute(
                "RangeSensorCreateLidar",
                path=env_lidar_path,
                min_range=0.4,
                max_range=100.0,
                draw_points=True,
                draw_lines=False,
                horizontal_fov=180.0,
                vertical_fov=30.0,
                horizontal_resolution=0.4,
                vertical_resolution=4.0,
                rotation_rate=0.0,
                high_lod=False,
                yaw_offset=0.0,
                enable_semantics=False,
            )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        action_scale = 10.0
        self.actions = actions.clone() * action_scale

    def _apply_action(self) -> None:
        self.carter.set_joint_velocity_target(self.actions, joint_ids=self._wheels_idx)
    
    def _get_lidar_data(self, lidar_interface):
        """Get LIDAR data from the simulation."""

        expected_points = int(180.0/0.4) # 180 degrees with 0.4 degree resolution
        all_lidar_data = torch.ones((self.num_envs, expected_points), device=self.device) * self.cfg.lidar_max_range
        
        for env_idx in range(self.num_envs):
            env_lidar_path = f"/World/envs/env_{env_idx}/Robot/chassis_link/carter_lidar"

            try:
                depth = lidar_interface.get_linear_depth_data(env_lidar_path)                
                
                if depth is not None and len(depth) > 0:
                    depth_tensor = torch.tensor(depth, device=self.device)
                    # Check if the depth data is valid
                    if len(depth_tensor) != expected_points:
                        if len(depth_tensor) < expected_points:
                            # Pad with max range values
                            padded = torch.ones(expected_points, device=self.device) * self.cfg.lidar_max_range
                            padded[:len(depth_tensor)] = depth_tensor
                            all_lidar_data[env_idx] = padded
                        else:
                            # Use middle section if we have more points than expected
                            middle_start = (len(depth_tensor) - expected_points) // 2
                            all_lidar_data[env_idx] = depth_tensor[middle_start:middle_start+expected_points] 
                else:
                    # Correct size
                    all_lidar_data[env_idx] = depth_tensor
            
            except Exception as e:
                # Print error for debugging
                print(f"Error getting LIDAR data for env {env_idx}: {e}")

        return all_lidar_data

    def _process_lidar_data(self, all_lidar_data):
        """Process LIDAR data to extract useful features for navigation."""
        
        num_envs = all_lidar_data.shape[0]
        num_points = all_lidar_data.shape[1]
        num_sectors = self.cfg.lidar_num_sectors
        max_range = self.cfg.lidar_max_range

        # Reset sector distances tensor
        self._sector_distances.fill_(max_range)
        
        # Calculate points per sector
        points_per_sector = num_points // num_sectors
        if points_per_sector == 0:
            points_per_sector = 1
            sectors_to_use = min(num_points, num_sectors)
        else:
            sectors_to_use = num_sectors

        # Process each environment using vectorized operations where possible
        for env_idx in range(num_envs):
            # Reshape data to handle processing in chunks
            # Handle the case where points aren't evenly divisible by sectors
            usable_points = points_per_sector * sectors_to_use
            if usable_points > num_points:
                usable_points = num_points - (num_points % sectors_to_use)

            # Create valid data mask (between min_range and max_range)
            valid_mask = (all_lidar_data[env_idx, :usable_points] > 0.4) & (all_lidar_data[env_idx, :usable_points] < max_range)
            valid_data = torch.where(valid_mask, all_lidar_data[env_idx, :usable_points], torch.tensor(max_range, device=self.device))

            # Process each sector
            for sector_idx in range(sectors_to_use):
                start_idx = sector_idx * points_per_sector
                end_idx = min(start_idx + points_per_sector, usable_points)

                if start_idx < end_idx:  # Ensure valid range
                    # Get minimum distance in this sector
                    sector_values = valid_data[start_idx:end_idx]
                    if len(sector_values) > 0:
                        min_val = torch.min(sector_values)

                        # Map 180° FOV to appropriate sectors in 360° view
                        # This mapping assumes the LIDAR faces forward and covers -90° to +90°

                        # Calculate actual sector in 360° view
                        # For 32 sectors, we map:
                        # - Far left (-90°) to sector 24
                        # - Front center (0°) to sector 0/31 (split)
                        # - Far right (+90°) to sector 8

                        front_center = 0  # Sector directly in front
                        half_sectors = num_sectors // 2
                        quarter_sectors = num_sectors // 4

                        # Linear mapping from sensor FOV to sector indices
                        normalized_pos = sector_idx / sectors_to_use  # 0.0 (left) to 1.0 (right)
                        angular_pos = normalized_pos * 180.0 - 90.0  # -90° (left) to +90° (right)

                        # Convert to sector index
                        if angular_pos < 0:  # Left side (-90° to 0°)
                            # Map to left quadrants (sectors from front_center to front_center+half_sectors)
                            sector_angle = 180.0 / half_sectors
                            angle_from_front = abs(angular_pos)
                            actual_sector_idx = front_center + int(angle_from_front / sector_angle)
                            actual_sector_idx = actual_sector_idx % num_sectors
                        else:  # Right side (0° to +90°)
                            # Map to right quadrants (sectors from front_center-1 down to front_center-quarter_sectors)
                            sector_angle = 90.0 / quarter_sectors
                            actual_sector_idx = front_center - 1 - int(angular_pos / sector_angle)
                            actual_sector_idx = (actual_sector_idx + num_sectors) % num_sectors

                        # Update the sector distance if this reading is closer
                        if min_val < self._sector_distances[env_idx, actual_sector_idx]:
                            self._sector_distances[env_idx, actual_sector_idx] = min_val
    
        # Normalize distances to [0, 1] range for neural network input
        normalized_distances = self._sector_distances / max_range
    
        return normalized_distances

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

        all_lidar_data = self._get_lidar_data(self.lidar_interface)
        processed_lidar = self._process_lidar_data(all_lidar_data)

        # Combine observations
        obs = torch.cat(
            (
                self._position_error.unsqueeze(dim=1),
                torch.cos(self.goal_heading_error).unsqueeze(dim=1),
                torch.sin(self.goal_heading_error).unsqueeze(dim=1),
                self.carter.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),  # Forward velocity
                self.carter.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),  # Lateral velocity
                self.carter.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),  # Angular velocity (yaw)
                processed_lidar,
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
            self.obstacle_penalty_weight,
            self.min_safe_distance,
            self._sector_distances,
            self.max_lidar_penalty_range,
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
    obstacle_penalty_weight: float,
    min_safe_distance: float,
    _sector_distances: torch.Tensor,
    max_lidar_penalty_range: float, 
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

    relevant_distances = torch.where(
        _sector_distances < max_lidar_penalty_range,
        _sector_distances,
        torch.full_like(_sector_distances, max_lidar_penalty_range),
    )
    
    # Option 1: Linear penalty below safe distance
    # proximity_penalty_term = torch.clamp(min_safe_distance - relevant_distances, min=0.0)

    # Option 2: Inverse penalty (more aggressive, be careful with tuning weight)
    proximity_penalty_term = torch.clamp(min_safe_distance / torch.clamp(relevant_distances, min=0.1), min=0.0, max=10.0) # Avoid division by zero and cap penalty

    obstacle_proximity_penalty = torch.sum(proximity_penalty_term, dim=-1)

    composite_reward = (
        position_progress_reward*position_progress_weight +
        goal_heading_reward*heading_progress_weight +
        goal_reached*goal_reached_reward -
        obstacle_proximity_penalty*obstacle_penalty_weight
    )

    if torch.any(composite_reward.isnan()):
        raise ValueError("Rewards cannot be NAN")
    
    #print("Position Progress:", position_progress_reward)
    #print("Heading Reward:", goal_heading_reward)
    #print("Goal Reached:", goal_reached)
    #print("Obstacle Penalty Term:", proximity_penalty_term)
    #print("Obstacle Penalty:", obstacle_proximity_penalty)
    
    return composite_reward, task_completed, _goal_index
