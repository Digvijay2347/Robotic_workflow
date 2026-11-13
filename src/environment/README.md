# Environment ROS 2 Package (`environment`)

This package provides a high-fidelity, dimensionally accurate URDF and Xacro model of the physical robotics workstation table with aluminum extrusion framing, transparent safety enclosures, solid robot mounting plate, overhead vision & lighting gantry, and peripheral equipment (UR controller box, teach pendant).

All dimensions and mounting offsets are modeled directly from the physical lab measurements.

---

## 1. Physical Specifications & Measurements

Based on physical measurements from the cell:

| Component | Dimensions / Measurement | Notes |
| :--- | :--- | :--- |
| **Tabletop Length ($X$)** | **$1.17\text{ m}$** ($117\text{ cm}$) | Total width along long edge ($39\text{ cm} + 42\text{ cm} + 37\text{ cm} \approx 1.18\text{ m}$) |
| **Tabletop Depth ($Y$)** | **$0.72\text{ m}$** ($72\text{ cm}$) | Front-to-back table width |
| **Table Surface Height ($Z$)** | **$0.78\text{ m}$** ($78\text{ cm}$) | From floor to top of table surface ($77\text{ cm}$ frame + feet) |
| **Aluminum Extrusions** | **$40\text{ mm} \times 40\text{ mm}$** ($0.04\text{ m}$ profile) | T-slot structural framing |
| **Robot Mounting Plate Size** | **$42\text{ cm} \times 30\text{ cm}$** ($0.42\text{ m} \times 0.30\text{ m}$) | Machined aluminum mounting plate |
| **Robot Plate Thickness** | **$2\text{ cm}$** ($0.02\text{ m}$) | Solid plate flush with rear edge |
| **Robot Mount Plate Offset** | **$39\text{ cm}$** from left edge | Centers plate near table midline |
| **Robot Base Mounting Center** | **$15\text{ cm}$** from rear edge | Centered on $30\text{ cm}$ plate width |
| **Overhead Gantry Height** | **$1.17\text{ m}$** above tabletop | Total height: $0.78\text{ m} + 1.17\text{ m} = 1.95\text{ m}$ |
| **Overhead Cantilever Overhang** | **$54\text{ cm}$** ($0.54\text{ m}$) | Extends over workspace |
| **Overhead Lighting** | **$1.05\text{ m}$** dual LED fixture | Suspended above table |
| **Teach Pendant** | **$0.32\text{ m} \times 0.24\text{ m}$** | Mounted on left extrusion, screen strictly parallel to side acrylic panel ($Y$-$Z$ plane) |
| **Enclosure** | **Clear acrylic / polycarbonate** | Side panels & front hinged doors with handles |
| **Lab Wall (Behind Table)** | **$2.40\text{ m} \times 2.50\text{ m} \times 0.10\text{ m}$** | Positioned with exact **$10\text{ cm}$ gap** from rear table edge ($Y = +0.46\text{ m}$ to $+0.56\text{ m}$) |

---

## 2. Coordinate Frames & Attachments

```
world
  └── table_base_link (Floor origin, Z=0)
        ├── table_frame_link (4040 aluminum extrusions & caster feet)
        ├── table_top_link (ESD mat work surface at Z = 0.78m)
        │     └── table_surface_frame (Tabletop center frame)
        ├── table_robot_mount_plate_link (Solid aluminum 20mm plate)
        │     └── table_robot_mount_frame / ur3_mount_link (Z = 0.80m, mounting point for robot)
        ├── table_enclosure_link (Transparent acrylic panels & front doors)
        ├── table_lower_shelf_link (Lower storage level at Z = 0.10m)
        │     └── ur_control_box_link (UR CB-series/e-series controller box)
        ├── overhead_gantry_link (1.17m vertical uprights & 54cm cantilever)
        │     ├── overhead_lighting_link (Dual LED lights)
        │     └── camera_overhead_mount_link (Magic arm & camera body)
        │           └── camera_overhead_optical_frame (Camera pointing down at table)
        ├── table_wall_link (Lab wall with 10cm gap behind table, Z=0 to 2.50m)
        └── teach_pendant_mount_link (Angled bracket holding UR PolyScope pendant)
```

---

## 3. Usage & Integration

### A. Including the Table in any Robot Xacro Description

You can attach the table into any URDF or Xacro model using the modular macro:

```xml
<xacro:include filename="$(find environment)/urdf/table_macro.xacro"/>

<!-- Instantiate the workstation table at world -->
<xacro:table prefix="" parent="world">
  <origin xyz="0 0 0" rpy="0 0 0"/>
</xacro:table>

<!-- Attach your robot base directly to the measured robot mount frame -->
<joint name="robot_to_table_mount_joint" type="fixed">
  <parent link="table_robot_mount_frame"/>
  <child link="base_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>
```

### B. Launching in RViz2

#### 1. View the Standalone Table
```bash
ros2 launch environment view_table.launch.py
```
Optional toggles:
```bash
ros2 launch environment view_table.launch.py include_overhead_gantry:=true include_enclosure:=true
```

#### 2. View the Complete UR3 + RG2 Gripper + Table Cell
```bash
ros2 launch environment view_table_with_robot.launch.py
```

---

## 4. Collision Geometry for MoveIt & Motion Planning

All components feature accurate `<collision>` boxes:
- Full tabletop clearance boundary ($Z = 0.78\text{ m}$, mounting plate $Z = 0.80\text{ m}$).
- Overhead lighting & gantry obstacle boundaries to prevent collisions during high pick-and-place trajectories.
- Lower cabinet enclosure boundary to safeguard against any lower sweeps.
