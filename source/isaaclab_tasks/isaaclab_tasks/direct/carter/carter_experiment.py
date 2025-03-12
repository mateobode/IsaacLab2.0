# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
import math
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##

def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/ground", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origin = [0.0, 0.0, 0.0]

    # Origin 1
    prim_utils.create_prim("/World/Origin", "Xform", translation=origin)
    # Origin 2
    #prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    ##
    # Configuration - Actuators
    ##
    
    CARTER_V1_ACTUATOR_CFG = ImplicitActuatorCfg(
        joint_names_expr=["left_wheel", "right_wheel"],
        effort_limit=40000.0,
        velocity_limit=100.0,
        stiffness=0.0,
        damping=1000.0,
    )

    CASTER_CFG = ImplicitActuatorCfg(
        joint_names_expr=["rear_pivot", "rear_axle"],
        effort_limit=0.0,
        velocity_limit=0.0,
        stiffness=0.0,
        damping=0.0,
    )
    ##
    # Configuration - Articulation
    ##
    
    CARTER_V1_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Carter/carter_v1_physx_lidar.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                #retain_accelerations=False,
                #linear_damping=0.1,
                #angular_damping=0.1,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=100.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=32,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05),
            rot=(0.0, 0.0, 0.0, 1.0),  # Ensure the robot is upright
            joint_pos={
                "left_wheel": 0.0,
                "right_wheel": 0.0,
                "rear_pivot": 0.0,
                "rear_axle": 0.0,
            },
        ),
        actuators={
            "wheels": CARTER_V1_ACTUATOR_CFG,
            "caster": CASTER_CFG,        },
    )
    
    """Configuration for the Carter V1 robot."""
    

    carter_cfg = CARTER_V1_CFG.copy()
    carter_cfg.prim_path = "/World/Origin/Robot"
    carter = Articulation(cfg=carter_cfg)
    # return the scene information
    return {"robot": carter}, origin


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origin: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origin
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos, 
                robot.data.default_joint_vel
                )
            # clear internal buffers
            robot.reset()
            print("Joints: ", robot.data.joint_names)
        # Apply differential control
        target_velocities = torch.zeros_like(robot.data.joint_pos)
        target_velocities[:, 0] = 1.0 # Left wheel
        target_velocities[:, 1] = 1.0 # Right wheel
        print(f"Target velocities: {target_velocities}")
        robot.set_joint_velocity_target(target_velocities)

        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)
        


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()