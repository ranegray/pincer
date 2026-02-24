```
 ____  ___ _   _  ____ _____ ____
|  _ \|_ _| \ | |/ ___| ____|  _ \
| |_) || ||  \| | |   |  _| | |_) |
|  __/ | || |\  | |___| |___|  _ <
|_|   |___|_| \_|\____|_____|_| \_\
```

a vision-to-grasping pipeline for low-cost robotic hardware.

pincer is a personal project focused on building reliable perception-to-manipulation on affordable hardware. it's also a hands-on application of my cs degree: inverse kinematics, computer vision, real-time control, and all the messy bits in between.

## current hardware

- **so101 arm** - 5-dof manipulator with feetech sts3215 servos
- **intel realsense d435if** - depth camera for 3d perception
- **3d-printed head/neck mount** - camera positioning
- **jetson orin nano super** - onboard compute

this is bus 1 from the [xlerobot](https://github.com/huggingface/lerobot/blob/main/examples/robots/xlerobot/README.md) setup.

## what's here

everything lives under `src/pincer/`.

**inverse kinematics** (`src/pincer/ik/`)

- `solver.py` - pink/pinocchio ik solver with convergence detection and early exit
- `model.py` - builds a reduced arm-only pinocchio model from the urdf with calibrated joint limits
- `transforms.py` - camera-to-base frame rigid-body transforms using forward kinematics
- `conventions.py` - motor <-> urdf angle convention mappings for arm and head joints
- `constants.py` - urdf paths, frame names, motor/joint name lists

**detection** (`src/pincer/detect/`)

- `color.py` - hsv-based color blob detection with configurable thresholds for red, green, and blue targets

**robot utilities** (`src/pincer/robots/`)

- `xlerobot_motor_utils.py` - motor bus helpers: pid configuration, calibration-based limit computation, position reading, and joint clipping

**camera** (`src/pincer/cameras/`)

- `d435.py` - realsense d435 wrapper for color + depth capture and 3d deprojection

**scripts** (`src/pincer/scripts/`)

| script | what it does |
|--------|--------------|
| `detect_and_grasp.py` | end-to-end pipeline: detect a colored object, solve ik, reach, grasp, and lift |
| `point_and_reach.py` | reach to the center-pixel depth target using camera-to-base transform and ik |
| `measure_ee.py` | manually position the gripper and read the model's end-effector position |
| `smoke_loop_numpy.py` | ik control loop using a numerical jacobian and damped least squares, pure numpy |
| `smoke_loop_pink.py` | ik control loop using pink/pinocchio for task-based inverse kinematics |
| `test_single_arm_limits.py` | motor calibration and limit testing |
| `smoke_bus1.py` | basic motor connectivity check |
| `read_bus1_live.py` | live motor position streaming |
| `smoke_realsense.py` | realsense depth camera smoke test |

**robot models** (`src/pincer/assets/`)

- urdf models for the xlerobot
- stl meshes for visualization and collision

## built on

this project is built on top of [lerobot](https://github.com/huggingface/lerobot) by hugging face, a fantastic open-source robotics framework that provides motor drivers, camera interfaces, dataset tooling, and state-of-the-art policy implementations. the hardware platform is the [xlerobot](https://github.com/huggingface/lerobot/blob/main/examples/robots/xlerobot/README.md).

additional libraries making this possible:

- [pinocchio](https://github.com/stack-of-tasks/pinocchio) - rigid body dynamics and kinematics
- [pink](https://github.com/stephane-caron/pink) - task-based inverse kinematics built on pinocchio
- [opencv](https://opencv.org/) - image processing and color-based object detection
