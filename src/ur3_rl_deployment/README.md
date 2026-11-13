# UR3 Reinforcement Learning Deployment Package for Jetson Orin Nano (ROS 2 Humble)

This package deploys the trained IsaacLab UR3 Reach Policy (`model_1499.pt` / `policy.onnx`) onto a Universal Robots UR3 (CB3.1 Controller running PolyScope 3.15+) via an NVIDIA Jetson Orin Nano.

---

## 1. Directory Structure

```
ur3_rl_deployment/
├── config/
│   ├── default_ur3_calibration.yaml   # Nominal kinematics parameters (ready to use)
│   └── policy_params.yaml             # Inference, safety, and topic parameters
├── launch/
│   └── ur3_rl_bringup.launch.py       # Brings up UR driver + policy node
├── models/
│   └── policy.onnx                    # Trained reaching policy ONNX model
├── resource/
│   └── ur3_rl_deployment              # ROS 2 ament resource marker
├── ur3_rl_deployment/
│   ├── __init__.py
│   ├── ur3_rl_policy_node.py          # 30 Hz closed-loop policy node
│   └── target_pose_publisher.py       # CLI tool to publish reach coordinates
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

---

## 2. Jetson Orin Nano Setup Instructions

### Step 1: Install System Dependencies
On the Jetson Orin Nano (Ubuntu 22.04 LTS / JetPack 6.x):
```bash
# 1. Install ROS 2 Humble base and UR Driver
sudo apt update
sudo apt install -y \
    ros-humble-ros-base \
    ros-humble-ur-robot-driver \
    ros-humble-ur-description \
    python3-pip \
    python3-colcon-common-extensions

# 2. Install ONNX Runtime with GPU support
pip3 install onnxruntime-gpu
```

### Step 2: Set up ROS 2 Workspace
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Copy this ur3_rl_deployment folder into ~/ros2_ws/src/
# Build the workspace
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

---

## 3. Network Configuration

1. Connect a direct Ethernet cable between Jetson Orin Nano and the UR CB3.1 Controller.
2. In Ubuntu (Jetson):
   - IP: `192.168.1.100`
   - Subnet Mask: `255.255.255.0`
   - Gateway: Empty
3. On PolyScope Pendant:
   - Setup Robot -> Network:
     - IP: `192.168.1.10`
     - Subnet Mask: `255.255.255.0`
4. Verify connection:
   ```bash
   ping 192.168.1.10
   ```

---

## 4. UR3 Calibration File (`my_ur3_calibration.yaml`)

- A nominal kinematics calibration file is included out-of-the-box:
  `config/default_ur3_calibration.yaml`
- **Optional (for sub-millimeter precision)**: Extract your physical arm's specific factory calibration directly from the controller:
  ```bash
  ros2 launch ur_calibration calibration_correction.launch.py \
      robot_ip:=192.168.1.10 \
      target_filename:="${HOME}/ros2_ws/src/ur3_rl_deployment/config/my_ur3_calibration.yaml"
  ```

---

## 5. Running the Deployment

### Phase A: Dry-Run (Verifying Everything Without Moving Motors)
```bash
ros2 launch ur3_rl_deployment ur3_rl_bringup.launch.py \
    robot_ip:=192.168.1.10 \
    enable_execution:=false
```
Watch the terminal output: verify that joint states are being received and policy inference outputs small safe incremental deltas.

### Phase B: Live Autonomous Execution
1. On the PolyScope Teach Pendant, jog the robot close to its default pose:
   `[0, -90, 90, -90, -90, 0]` degrees.
2. Start the `External Control` program on the pendant.
3. Launch with live execution enabled:
   ```bash
   ros2 launch ur3_rl_deployment ur3_rl_bringup.launch.py \
       robot_ip:=192.168.1.10 \
       enable_execution:=true
   ```

### Phase C: Send Reaching Targets

You have two options to command targets:

#### Option 1: MoveIt-Style 3D Interactive Marker in RViz (Recommended GUI)
When launching `ur3_rl_bringup.launch.py` (which launches RViz via `ur_control` by default, or with `launch_rviz:=true`), the `interactive_target_publisher` node automatically starts.

**Key Features:**
- **Auto-Sync to Robot EE**: On startup, the marker queries TF and spawns **directly on the robot's current end-effector pose** (Home position).
- **Safe "Lock & Send" Workflow**: Target pose is **LOCKED while moving**. You can freely drag and adjust the marker as much as you want without the robot moving. When satisfied, click the floating 3D `[ 🔒 LOCK & SEND ]` button in RViz or select it from the right-click menu to dispatch the command!
- **Free Multi-Mode Dragging**:
  - Click & drag the **center target sphere** to move freely in the camera view plane.
  - Drag the **planar handles** to slide along XY, XZ, or YZ planes.
  - Drag the **red, green, blue arrows** for single-axis translation.
  - Rotate the **rings** to adjust end-effector roll, pitch, and yaw.
- **360° Workshell Reach**: Supports full 360° spherical/cylindrical reach around the robot base (`Radius: [0.10, 0.50]m`, `Z: [0.02, 0.50]m`).

Inside the RViz window that opened:
1. In the **Displays** panel on the left, click **Add** (bottom left).
2. Select **Interactive Markers** and click **OK**.
3. Under **Interactive Markers** in the list, set **Update Topic** to:
   ```text
   /target_pose_marker/update
   ```
4. **To view the 3D Workspace Bounding Box**:
   - **Using standard "Marker" display**: Click **Add** -> **Marker**, and in the Topic dropdown select:
     ```text
     /ur3_reach/workspace_box_marker
     ```
     *(or `/ur3_reach/bounding_box_marker`)* — displays the clean 12-edge cyan 3D wireframe bounding box.
   - **Using "MarkerArray" display**: Click **Add** -> **MarkerArray**, and set the topic to:
     ```text
     /ur3_reach/workspace_bounds_marker
     ```
     *(Displays the full workspace: 3D bounding box wireframe, translucent volume, reach cylinders, rings, and coordinate label)*.
5. **Adjust & Send**:
   - Drag the sphere, planar handles, or arrows to place your desired target.
   - Click the green **`[ 🔒 LOCK & SEND ]`** 3D button directly above the marker (or right-click -> **Lock & Send Target Pose**) to send the goal to the UR3.

#### Option 2: Command Line (CLI)
If you prefer numeric CLI commands without dragging in RViz:
```bash
# Send reach target (X=0.35m, Y=0.08m, Z=0.25m)
ros2 run ur3_rl_deployment target_pose_publisher --x 0.35 --y 0.08 --z 0.25
```

