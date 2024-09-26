"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob
from tqdm import tqdm

import h5py
import numpy as np
import matplotlib.pyplot as plt

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
import mimicgen

AXIS_IDX = {
        'rx': 3,
        'ry': 4,
        'rz': 5,
    }
# ACTION_FREQ = 0.2

def toggle_rotation(env, num_of_toggle:int, axis:str, render=False):
    assert axis in ['rx', 'ry', 'rz']
    rot = -1. if num_of_toggle < 0 else 1.
    for _ in range(abs(num_of_toggle)):
        gripper = np.array([np.random.uniform(-1., 1.)]).astype(float)
        action = np.array([0., 0., 0., 0., 0., 0.])
        action = np.concatenate([action, gripper])
        action[AXIS_IDX[axis]] = rot
        env.step(action)
        # print(env.robots[0].recent_ee_pose.last)
        if render:
            env.render()
        # time.sleep(ACTION_FREQ)

def rotation_backforth(env, r_min:int, r_max:int, axis:str):
    assert axis in ['rx', 'ry', 'rz']
    toggle_rotation(env, r_min, axis)
    toggle_rotation(env, -r_min, axis)
    toggle_rotation(env, r_max, axis)
    toggle_rotation(env, -r_max, axis)

def collect_organized_spatial_trajectory(env, arm, env_configuration, spatial_xyz_goal, spatial_resolution, render=False):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()

    # ID = 2 always corresponds to agentview
    if render:
        env.render()

    low, high = env.action_spec
    gripper_low, gripper_high = low[-1], high[-1]
    
    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal

    # Go to xyz goal
    i = 0
    while i < 300:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]
        current_robot_xyz = active_robot.recent_ee_pose.last[:3]
        action_xyz = (spatial_xyz_goal - current_robot_xyz) / np.linalg.norm(spatial_xyz_goal - current_robot_xyz) #* 3.75
        action_rxyz = np.array([0., 0., 0.])
        gripper = np.array([np.random.uniform(gripper_low, gripper_high)]).astype(float)
        action = np.concatenate([action_xyz, action_rxyz, gripper])
        
        # time.sleep(ACTION_FREQ)
        # print(active_robot.recent_ee_pose.last)
        env.step(action)
        if render:
            env.render()
        # if i % 100 ==0:
        #     print(env.sim.data.qpos[env.drawer_qpos_addr])
        # Also break if we complete the task
        if np.linalg.norm(spatial_xyz_goal - current_robot_xyz) < spatial_resolution:
            break
        i += 1
    # do x-axis-only rotations
        
    # rz_min, rz_max = -30, 53
    rz_min, rz_max = -20, 45
    rotation_backforth(env, rz_min, rz_max, 'rz', render=render)
    # rx_min, rx_max = -22, 22
    rx_min, rx_max = -5, 5
    rotation_backforth(env, rx_min, rx_max, 'rx', render=render)
    # ry_min, ry_max = -17, 10
    ry_min, ry_max = -5, 5
    rotation_backforth(env, ry_min, ry_max, 'ry', render=render)
    
    # do some random movement around the goal xyz
    
    # do visualization
    # for i in range(200):
    #     action = np.random.uniform(low, high)
    #     obs, reward, done, _ = env.step(action)
    #     if render:
    #         env.render()
    # cleanup for end of data collection episodes
        
    env.close()

def collect_random_spatial_trajectory(env, arm, env_configuration, spatial_xyz_goal, spatial_resolution, render=False):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()

    # ID = 2 always corresponds to agentview
    if render:
        env.render()

    low, high = env.action_spec
    gripper_low, gripper_high = low[-1], high[-1]
    
    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal

    # Go to xyz goal
    if spatial_xyz_goal is not None:
        while True:
            # Set active robot
            active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]
            current_robot_xyz = active_robot.recent_ee_pose.last[:3]
            action_xyz = (spatial_xyz_goal - current_robot_xyz) / np.linalg.norm(spatial_xyz_goal - current_robot_xyz) #* 3.75
            action_rxyz = np.array([0., 0., 0.])
            gripper = np.array([np.random.uniform(gripper_low, gripper_high)]).astype(float)
            action = np.concatenate([action_xyz, action_rxyz, gripper])
            
            # time.sleep(ACTION_FREQ)
            print(active_robot.recent_ee_pose.last)
            env.step(action)
            if render:
                env.render()
            # if i % 100 ==0:
            #     print(env.sim.data.qpos[env.drawer_qpos_addr])
            # Also break if we complete the task
            if np.linalg.norm(spatial_xyz_goal - current_robot_xyz) < spatial_resolution:
                break
    # do some random movement around the goal xyz
    
    # do visualization
    for i in range(200):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        if render:
            env.render()
    # cleanup for end of data collection episodes
        
    env.close()

def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset

        print("Demonstration is successful and has been saved")
        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="spaceview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument('--enable_render', action='store_true')
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    enable_render = args.enable_render
    if enable_render:
        env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)


    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # spatial_resolution = 0.013
    spatial_resolution = 0.05
    x_min, x_max = -0.4, 0.2
    x_ = np.linspace(x_min, x_max, int((x_max-x_min)//spatial_resolution))
    y_min, y_max = -0.4, 0.4
    y_ = np.linspace(y_min, y_max, int((y_max-y_min)//spatial_resolution))
    z_min, z_max = 0.82, 1.2
    z_ = np.linspace(z_min, z_max, int((z_max-z_min)//spatial_resolution))
    xyz_coordinates = np.vstack(np.meshgrid(x_,y_,z_)).reshape(3,-1).T
    print(f"there arr {xyz_coordinates.shape[0]} 3d points to be done")
    # collect demonstrations
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(xyz_coordinates[:,0], xyz_coordinates[:,1], xyz_coordinates[:,2])
    # plt.show()
    
    for xyz in tqdm(xyz_coordinates, desc="generate organized spatial trajectory"):
        # xyz = np.array([0,0,1.0])
        collect_organized_spatial_trajectory(env, args.arm, args.config, xyz, spatial_resolution=np.sqrt(spatial_resolution**2*3), render=enable_render)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
        break
    for xyz in tqdm(xyz_coordinates, desc="generate random spatial trajectory"):
        # xyz = np.array([0,0,1.0])
        collect_random_spatial_trajectory(env, args.arm, args.config, xyz, spatial_resolution=np.sqrt(spatial_resolution**2*3), render=enable_render)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
    for xyz in tqdm(range(100), desc="generate random spatial trajectory at start"):
        goal = None
        collect_random_spatial_trajectory(env, args.arm, args.config, goal, spatial_resolution=np.sqrt(spatial_resolution**2*3), render=enable_render)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)