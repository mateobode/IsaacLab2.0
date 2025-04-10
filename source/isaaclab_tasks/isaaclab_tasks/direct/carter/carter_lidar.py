# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025, Mateo Bode Nakamura Lab.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from collections.abc import Sequence

from .carter import CARTER_V1_CFG
from .goal import WAYPOINT_CFG
from .walls import WALL_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaacsim.sensors.physx import _range_sensor


@configclass
class CarterEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2 # Decimation factor for rendering
    episode_length_s = 30.0 # Maximum episode length in seconds
    action_space = 2 # Number of actions the neural network should return (wheel velocities)
    # Number of observations fed to the neural network
    observation_space = 16
    state_space = 0

    env_spacing = 16.0 # Spacing between environments, depends on the amount of goals
    num_goals = 10 # Number of goals in the environment

    course_length_coefficient = 2.5 # Coefficient for the length of the course
    course_width_coefficient = 2.0 # Coefficient for the width of the course

    # simulation frames Hz
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CARTER_V1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    wheels = ["left_wheel", "right_wheel"]
    caster = ["rear_pivot", "rear_axle"]

    # goal waypoints
    waypoint_cfg: VisualizationMarkersCfg = WAYPOINT_CFG

    # walls
    wall_cfg = []
    for i in range(num_goals):
        wall = WALL_CFG.copy()
        wall.prim_path = f"/World/envs/env_.*/Wall{i}"
        wall_cfg.append(wall)

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
        self._wall_positions = torch.zeros((self.num_envs, self._num_goals, 2), dtype=torch.float32, device=self.device)
        self._goal_index = torch.zeros((self.num_envs), dtype=torch.int32, device=self.device)
        self._marker_position = torch.zeros((self.num_envs, self._num_goals, 3), dtype=torch.float32, device=self.device)
        
        try:
            self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
            print("LiDAR interface successfully initialized")
        except Exception as e:
            print(f"Error initializing LiDAR interface: {e}")
            # Provide a fallback
            self.lidar_interface = None
    
        self._closest_obstacle = torch.ones((self.num_envs), dtype=torch.float32, device=self.device) * 10.0

        # Reward coefficients
        self.linear_velocity_min = 0.0
        self.linear_velocity_max = 1.0
        self.angular_velocity_min = 0.0
        self.angular_velocity_max = 1.0

        # Flag to skip the first few frames of LiDAR data (might be unstable)
        self._lidar_warmup_frames = 10
        self._current_frame = 0
        
        # Schedule a delayed debug call after initialization
        print("Environment initialized. Will debug LiDAR sensor after first reset.")

    def _setup_scene(self):
        self.carter = Articulation(self.cfg.robot_cfg)
        
        # add goal waypoints
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)

        # add walls
        self.walls = []

        for wall in self.cfg.wall_cfg:
            self.walls.append(RigidObject(wall))

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
        # lidar path
        lidar_path = "/World/envs/env_.*/Robot/chassis_link/carter_lidar"

        # Get current goal position
        current_goal_position = self._goal_positions[self.carter._ALL_INDICES, self._goal_index]

        # Calculate position error vector and magnitude
        position_error_vector = current_goal_position - self.carter.data.root_pos_w[:, :2]
        position_error = torch.norm(position_error_vector, dim=-1)

        current_wall_positions = self._wall_positions[self.carter._ALL_INDICES, self._goal_index]

        # Calculate heading error
        heading = self.carter.data.heading_w
        target_heading_w = torch.atan2(
            current_goal_position[:, 1] - self.carter.data.root_pos_w[:, 1],
            current_goal_position[:, 0] - self.carter.data.root_pos_w[:, 0]
        )
        heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        NUM_SECTORS = 8
        lidar_sectors = torch.ones((self.num_envs, NUM_SECTORS), dtype=torch.float32, device=self.device) * 10.0

        self._closest_obstacle = torch.ones((self.num_envs), dtype=torch.float32, device=self.device) * 10.0

        # Process LiDAR data for each environment
        for i in range(self.num_envs):
            try:
                # Get LiDAR path for this specific environment
                env_lidar_path = f"/World/envs/env_{i}/Robot/chassis_link/carter_lidar"

                # Get raw LiDAR data (depth only for now)
                depth_data = self.lidar_interface.get_linear_depth_data(env_lidar_path)

                # Process depth data if available
                if depth_data is not None and len(depth_data) > 0:
                    # Convert to PyTorch tensor
                    depth_np = np.array(depth_data)

                    # Print the shape and content of depth_data for debugging
                    if i == 0:  # Only print for first environment to avoid spam
                        print(f"LiDAR data shape: {depth_np.shape}, Type: {type(depth_np)}")
                        if len(depth_np) > 0:
                            print(f"Sample values: {depth_np[:5]}")

                    # Filter out invalid readings (too far)
                    valid_mask = depth_np < 10.0
                    if np.any(valid_mask):
                        valid_depths = depth_np[valid_mask]
                        self._closest_obstacle[i] = float(np.min(valid_depths))

                    # For simplicity, just divide the readings into 8 equal sectors
                    # by azimuth angle (we assume the data is organized that way)
                    if len(depth_np) >= NUM_SECTORS:
                        # Simplification: Just take every N-th reading
                        step = len(depth_np) // NUM_SECTORS
                        for j in range(NUM_SECTORS):
                            idx = j * step
                            if idx < len(depth_np):
                                lidar_sectors[i, j] = float(depth_np[idx])
                    else:
                        # Not enough readings, just use what we have
                        for j in range(min(NUM_SECTORS, len(depth_np))):
                            lidar_sectors[i, j] = float(depth_np[j])

            except Exception as e:
                # Print detailed error for debugging
                import traceback
                print(f"Error processing LiDAR for env {i}: {str(e)}")
                traceback.print_exc()


        # Combine observations
        obs = torch.cat(
            (
                position_error.unsqueeze(dim=1),
                torch.cos(heading_error).unsqueeze(dim=1),
                torch.sin(heading_error).unsqueeze(dim=1),
                current_wall_positions[:, 0].unsqueeze(dim=1),
                current_wall_positions[:, 1].unsqueeze(dim=1),
                self.carter.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),  # Forward velocity
                self.carter.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),  # Lateral velocity
                self.carter.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),  # Angular velocity (yaw)
                lidar_sectors,
            ),
            dim=-1
        )

        # Update the observation space to match
        self.cfg.observation_space = obs.shape[1]

        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        observations = {"policy": obs}
        return observations

    def debug_lidar_sensor(self, env_id=0):
        """Debug function to understand the LiDAR sensor data structure."""
        # Get LiDAR path for the specified environment
        env_lidar_path = f"/World/envs/env_{env_id}/Robot/chassis_link/carter_lidar"

        print(f"\n=== LiDAR Debug for Environment {env_id} ===")
        print(f"LiDAR path: {env_lidar_path}")

        try:
            # Check if the LiDAR interface is available
            if not hasattr(self, 'lidar_interface') or self.lidar_interface is None:
                print("ERROR: LiDAR interface not initialized")
                return

            # Try to access different types of LiDAR data
            depth_data = self.lidar_interface.get_linear_depth_data(env_lidar_path)
            azimuth_data = self.lidar_interface.get_azimuth_data(env_lidar_path)
            zenith_data = self.lidar_interface.get_zenith_data(env_lidar_path)

            # Print information about each data type
            print("\n--- Depth Data ---")
            if depth_data is not None:
                print(f"Type: {type(depth_data)}")
                print(f"Shape: {np.shape(depth_data)}")
                print(f"Size: {len(depth_data)}")
                if len(depth_data) > 0:
                    print(f"Range: {np.min(depth_data)} to {np.max(depth_data)}")
                    print(f"First 5 values: {depth_data[:5]}")
            else:
                print("No depth data available")

            print("\n--- Azimuth Data ---")
            if azimuth_data is not None:
                print(f"Type: {type(azimuth_data)}")
                print(f"Shape: {np.shape(azimuth_data)}")
                print(f"Size: {len(azimuth_data)}")
                if len(azimuth_data) > 0:
                    print(f"Range: {np.min(azimuth_data)} to {np.max(azimuth_data)}")
                    print(f"First 5 values: {azimuth_data[:5]}")
            else:
                print("No azimuth data available")

            print("\n--- Zenith Data ---")
            if zenith_data is not None:
                print(f"Type: {type(zenith_data)}")
                print(f"Shape: {np.shape(zenith_data)}")
                print(f"Size: {len(zenith_data)}")
                if len(zenith_data) > 0:
                    print(f"Range: {np.min(zenith_data)} to {np.max(zenith_data)}")
                    print(f"First 5 values: {zenith_data[:5]}")
            else:
                print("No zenith data available")

            # Try getting other potential methods/properties of the LiDAR interface
            print("\n--- LiDAR Interface Methods ---")
            interface_methods = [method for method in dir(self.lidar_interface) 
                                 if callable(getattr(self.lidar_interface, method)) and not method.startswith('_')]
            print(f"Available methods: {interface_methods}")

        except Exception as e:
            import traceback
            print(f"ERROR during LiDAR debugging: {str(e)}")
            traceback.print_exc()

        print("=== End of LiDAR Debug ===\n")

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
        
        # Add collision avoidance penalty
        collision_threshold = 1.0  # Start penalizing when closer than 1 meter
        collision_penalty_weight = 50.0  # Adjust weight of penalty
        
        # Calculate penalty (increases as robot gets closer to obstacles)
        collision_penalty = torch.where(
            self._closest_obstacle < collision_threshold,
            -collision_penalty_weight * ((collision_threshold - self._closest_obstacle) / collision_threshold) ** 2,
            torch.zeros_like(self._closest_obstacle)
        )

        # Imminent collision penalty
        imminent_collision = self._closest_obstacle < 0.25
        collision_penalty = torch.where(
            imminent_collision,
            -200.0, # Max penalty for very close to obstacles
            collision_penalty
        )
        
        # Add collision avoidance penalty to composite reward
        composite_reward = composite_reward + collision_penalty
        
        # Update waypoint visualization
        one_hot_encoded = torch.nn.functional.one_hot(self._goal_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)
        
        composite_reward = composite_reward + collision_penalty

        return composite_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Task fails if episode length is exceeded
        failure_termination = self.episode_length_buf >= self.max_episode_length - 1

        # Collision detection termination
        collision_detected = self._closest_obstacle < 0.2
        failure_termination = failure_termination | collision_detected

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

        # Set up goal positions
        spacing = 2 / self._num_goals
        goal_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        self._goal_positions[env_ids, :len(goal_positions), 0] = goal_positions
        self._goal_positions[env_ids, :, 1] = torch.rand((num_reset, self._num_goals), dtype=torch.float32, device=self.device) * self.course_width_coefficient
        self._goal_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        # Reset goal index
        self._goal_index[env_ids] = 0

        # Reset markers
        self._marker_position[env_ids, :, :2] = self._goal_positions[env_ids]
        visualize_pos = self._marker_position.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)

        # Reset walls
        offset = 1.5
        self._wall_positions[env_ids] = self._goal_positions[env_ids]
        offset = torch.full((num_reset, self._num_goals), fill_value=offset, dtype=torch.float, device=self.device)
        sign_pattern = torch.tensor([1 if j%2 == 0 else -1 for j in range(self._num_goals)], device=self.device)
        offset[:, :] *= sign_pattern
        self._wall_positions[env_ids, :, 1] += offset

        index: int = 0
        for wall in self.walls:
            wall_default_state = wall.data.default_root_state[env_ids]
            wall_pose = wall_default_state[:, :7].clone()
            wall_pose[:, 2] = 0.25
            wall_pose[:, :2] = self._wall_positions[env_ids, index, :]
            
            index += 1

            wall.write_root_pose_to_sim(wall_pose, env_ids)
            wall_velocities = wall_default_state[:, 7:]
            wall.write_root_velocity_to_sim(wall_velocities, env_ids)

        current_goal_positions = self._goal_positions[self.carter._ALL_INDICES, self._goal_index]
        self._position_error_vector = current_goal_positions[:, :2] - self.carter.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()
        
        # reset heading error
        heading = self.carter.data.heading_w[:]
        target_heading_w = torch.atan2(
            self._goal_positions[:, 0, 1] - self.carter.data.root_pos_w[:, 1],
            self._goal_positions[:, 0, 0] - self.carter.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self._previous_heading_error = self._heading_error.clone()


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
