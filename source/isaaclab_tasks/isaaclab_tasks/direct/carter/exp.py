# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Carter Experiment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import omni

import isaaclab.sim as sim_utils

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaacsim.sensors.physx import _range_sensor
from carter import CARTER_V1_CFG


@configclass
class ContactSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = CARTER_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Rigid Object
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.0)),
    )

    carter_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis_link",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Get the LiDAR interface
    lidar_interface = _range_sensor.acquire_lidar_sensor_interface()

    # Simulate physics
    while simulation_app.is_running():

        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            #joint_vel += torch.rand_like(joint_vel) * 1.0
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        targets = torch.zeros_like(scene["robot"].data.joint_pos)
        targets[:, 0] = 1.0
        targets[:, 1] = 1.0
        # -- apply action to the robot
        scene["robot"].set_joint_velocity_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        print("-------------------------------")
        print(scene["carter_contact_sensor"])
        print("Received force matrix of: ", scene["carter_contact_sensor"].data.force_matrix_w)
        print("Received contact force of: ", scene["carter_contact_sensor"].data.net_forces_w)
        print("-------------------------------")

        # print lidar sensor data
        if count % 10 == 0:  # Only print every 10 steps to reduce console spam
            print("-------------------------------")
            # Get the actual LiDAR path by properly replacing the ENV_REGEX_NS placeholder
            lidar_path = None
            # Try to debug and find the actual LiDAR path
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            # First, try with an empty environment namespace (for single environment)
            test_path = "/World/Robot/chassis_link/carter_lidar"
            if stage.GetPrimAtPath(test_path).IsValid():
                lidar_path = test_path
                print(f"Found LiDAR at: {lidar_path}")
            else:
                # If not found, try with env_0 (common environment namespace)
                test_path = "/World/env_0/Robot/chassis_link/carter_lidar"
                if stage.GetPrimAtPath(test_path).IsValid():
                    lidar_path = test_path
                    print(f"Found LiDAR at: {lidar_path}")
                else:
                    # If still not found, search all possible paths
                    print("Searching for LiDAR path...")
                    # Try to find the actual path by listing all prims with "lidar" in their name
                    for prim in stage.Traverse():
                        if "lidar" in prim.GetPath().pathString.lower():
                            print(f"Found potential LiDAR at: {prim.GetPath()}")
                            lidar_path = str(prim.GetPath())
                            break
            
            if lidar_path:
                try:
                    # Try to get LiDAR data
                    depth_data = lidar_interface.get_linear_depth_data(lidar_path)
                    
                    print(f"Depth data: {depth_data}")
                except Exception as e:
                    print(f"Error accessing LiDAR data: {e}")
            else:
                print("Could not find LiDAR path. Please check the actual path in the scene.")
            print("-------------------------------")


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = ContactSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()