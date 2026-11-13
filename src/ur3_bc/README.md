# UR3 Behavioral Cloning (BC) ROS 2 Deployment Package (`ur3_bc`)

Production-ready ROS 2 package for deploying the 125 Hz closed-loop neural trajectory policy on the Universal Robots UR3 + OnRobot RG2 cell.

## Package Architecture
* **Target Frame**: Strictly **`base_link`** (100% robot-centric, completely portable).
* **Safety Geofence**: Validated in **`table_robot_mount_frame`** ($90\,\text{cm} \times 40\,\text{cm} \times 26\,\text{cm}$).
* **Inference Engine**: Dual-backend (`PyTorch` or standalone `NumPy`, $<0.2\,\text{ms}$ latency, zero GPU memory footprint).

---

## Quickstart

### 1. Build the Package
In your ROS 2 workspace:
```bash
cd /home/surender/ros2_venv/ur3_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ur3_bc --symlink-install
source install/setup.bash
```

---

### 2. Task 1: Reaching Target Poses (125 Hz Closed-Loop)

#### Step 1: Launch Robot Driver
```bash
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur3 \
    robot_ip:=172.17.0.2 \
    use_fake_hardware:=false \
    description_package:=environment \
    description_file:=ur3_with_table.urdf.xacro \
    initial_joint_controller:=forward_position_controller \
    launch_rviz:=false
```
*(On UR teach pendant: Play External Control program).*

#### Step 2: Launch Task 1 Node
```bash
ros2 launch ur3_bc task1_reach.launch.py
# Or directly via node:
# ros2 run ur3_bc task1_reach_target
```

#### Step 3: Command Target Poses
```bash
# Send workspace center target:
ros2 run ur3_bc send_target --center

# Send specific coordinate in base_link:
ros2 run ur3_bc send_target --x 0.25 --y 0.15 --z 0.10

# Interactive prompt:
ros2 run ur3_bc send_target
```

---

### 3. Task 2: Pick and Place State Machine

#### Step 1: Launch Robot Driver (if not already running)
Ensure `forward_position_controller` is active and External Control is playing.

#### Step 2: Run Task 2 Node
```bash
ros2 launch ur3_bc task2_pick_place.launch.py \
    pick_x:=0.28 pick_y:=0.12 pick_z:=0.04 \
    place_x:=0.28 place_y:=-0.15 place_z:=0.04
```
*(The robot will automatically approach from $+10\,\text{cm}$ above, open gripper, grasp, verify contact width via `/rg2_state`, lift, transport, place, release, and park at Ready Pose).*

---

### 4. Deploying to NVIDIA Jetson

To transfer to your Jetson:
```bash
# On your laptop, copy the package to Jetson:
rsync -avz /home/surender/ros2_venv/ur3_ws/src/ur3_bc user@<jetson_ip>:~/ur3_ws/src/

# On Jetson:
cd ~/ur3_ws
colcon build --packages-select ur3_bc
source install/setup.bash
```
Because `ur3_bc` includes a pure NumPy inference fallback, it will run immediately on Jetson even if PyTorch is not installed in the ROS 2 Python environment!
