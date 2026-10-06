# Environment Setup & Operational Execution Guide

**IsaacLab UR10e Bin-to-Bin Manipulation System**  
*Document Version: 2.0 (Latest Working Codebase)*  
*Focus: Environment Setup, Checkpoint Verification, Headless & GUI Evaluation, Pretraining, and Troubleshooting*

---

## 1. System Requirements & Prerequisites

### 1.1 Hardware Specifications
- **GPU**: NVIDIA RTX 3080 / 4090 / RTX A5000 / RTX 6000 Ada with at least 16 GB VRAM.
- **CPU**: x86_64 multi-core processor (8+ physical cores recommended).
- **RAM**: Minimum 32 GB system memory.
- **Disk**: 50 GB free disk space (for Isaac Sim assets, USD caches, and datasets).

### 1.2 Software Stack
- **Operating System**: Ubuntu 22.04 LTS (x86_64).
- **NVIDIA Driver**: $\ge 535.129.03$ (CUDA 12.x compatible).
- **Simulator Platform**: NVIDIA Isaac Sim 4.5.0.
- **Framework**: NVIDIA Isaac Lab 2.x.
- **Python**: Python 3.12 (managed in virtual environment).

---

## 2. Environment Configuration & Python Dependencies

The system uses a dedicated Python virtual environment located at `/home/azureuser/new_isaac/env_isaaclab`.

### 2.1 Tooling & Python Invocation Philosophy
As established in IsaacLab guidelines, run Python commands via the IsaacLab bash wrapper or directly using the virtual environment:
- **IsaacLab wrapper**: `./isaaclab.sh -p <script.py> [args]`
- **Virtualenv direct binary**: `/home/azureuser/new_isaac/env_isaaclab/bin/python <script.py> [args]`

### 2.2 Installing Python Dependencies
To ensure all perception and imitation learning libraries are available:

```bash
# Activate workspace directory
cd /home/azureuser/new_isaac/IsaacLab

# Install required deep learning and robotics packages
/home/azureuser/new_isaac/env_isaaclab/bin/pip install \
    ultralytics>=8.0.0 \
    stable-baselines3>=2.0.0 \
    gymnasium>=0.29.0 \
    h5py \
    opencv-python-headless
```

### 2.3 Smoke Test Verification
Verify that PyTorch, CUDA, Isaac Sim, and Ultralytics import cleanly without context collisions:

```bash
/home/azureuser/new_isaac/env_isaaclab/bin/python -c "
import torch
import ultralytics
from ultralytics import YOLO

print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
print('Ultralytics Version:', ultralytics.__version__)
"
```

---

## 3. Directory Structure & Weight Asset Checklist

Ensure all required model checkpoints, USD assets, and datasets are in their expected locations:

```
IsaacLab/
├── checkpoints/
│   ├── bin_transfer_bc_policy.pt            # [CRITICAL] M20 Bolt BC policy
│   ├── coupler_bin_transfer_bc_policy.pt    # [CRITICAL] Bolt Coupler BC policy
│   ├── yolov8n_bin_transfer_seg.pt          # [CRITICAL] Fine-tuned YOLOv8-Seg weights
│   └── overhead_object_seg_model.pt         # Optional lightweight seg net weights
│
├── datasets/
│   ├── bin_transfer_expert_demos.h5         # Bolt demonstration dataset
│   └── coupler_bin_transfer_expert_demos.h5 # Coupler demonstration dataset
│
└── usd/                                     # Local USD assets (bins, bolts, couplers)
```

### Integrity Verification Command
Run this command from the repository root to verify asset presence:

```bash
/home/azureuser/new_isaac/env_isaaclab/bin/python -c "
import os

required_files = [
    'checkpoints/bin_transfer_bc_policy.pt',
    'checkpoints/coupler_bin_transfer_bc_policy.pt',
    'checkpoints/yolov8n_bin_transfer_seg.pt',
    'scripts/reinforcement_learning/bin_transfer/evaluate_with_perception.py',
    'scripts/reinforcement_learning/bin_transfer/evaluate_coupler_with_perception.py'
]

missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    print('WARNING: Missing files:', missing)
else:
    print('SUCCESS: All critical checkpoints and evaluation scripts verified.')
"
```

---

## 4. Execution Runbook: Closed-Loop Perception Evaluations

### 4.1 Task 1: M20 Bolt Transfer (`evaluate_with_perception.py`)
Transfers an M20 industrial bolt from the source bin into the target bin under active RealSense RGB-D perception and 3-state occlusion latching.

#### Headless Evaluation (Fastest / CI / Cloud Server)
```bash
DISPLAY="" /home/azureuser/new_isaac/env_isaaclab/bin/python \
    scripts/reinforcement_learning/bin_transfer/evaluate_with_perception.py \
    --headless \
    --num_envs 1 \
    --num_episodes 5
```

#### GUI Real-Time Visualization (Interactive NVIDIA Kit Window)
```bash
/home/azureuser/new_isaac/env_isaaclab/bin/python \
    scripts/reinforcement_learning/bin_transfer/evaluate_with_perception.py \
    --viz kit \
    --real_time \
    --num_envs 1 \
    --num_episodes 5
```

---

### 4.2 Task 2: Bolt Coupler Transfer (`evaluate_coupler_with_perception.py`)
Transfers a cylindrical bolt coupler from the source bin into the target bin with symmetry-aware grasp yaw orientation.

#### Headless Evaluation
```bash
DISPLAY="" /home/azureuser/new_isaac/env_isaaclab/bin/python \
    scripts/reinforcement_learning/bin_transfer/evaluate_coupler_with_perception.py \
    --headless \
    --num_envs 1 \
    --num_episodes 5
```

#### GUI Real-Time Visualization
```bash
/home/azureuser/new_isaac/env_isaaclab/bin/python \
    scripts/reinforcement_learning/bin_transfer/evaluate_coupler_with_perception.py \
    --viz kit \
    --real_time \
    --num_envs 1 \
    --num_episodes 5
```

---

### 4.3 Key Command-Line Flags Explained

| Argument | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--headless` | Flag | `False` | Disables rendering window; runs simulation purely offscreen (ideal for remote cloud VMs). |
| `--viz kit` | Choice | None | Launches the Omniverse Kit viewport UI for visual inspection. |
| `--real_time` | Flag | `False` | Paces the simulation loop (~40 ms delay per step) to match human visual playback speed. |
| `--num_envs` | Int | `4` | Number of parallel vectorized environments to instantiate. |
| `--num_episodes` | Int | `20` | Total number of test episodes to evaluate before computing final metrics. |
| `--checkpoint` | Path | `checkpoints/...` | Path to the PyTorch `.pt` policy checkpoint to load. |
| `--pos_noise` | Float | `0.0025` | Standard deviation of simulated Gaussian camera position noise [m] ($2.5\,\text{mm}$). |
| `--yaw_noise` | Float | `0.025` | Standard deviation of simulated Gaussian grasp yaw noise [rad] ($\approx 1.4^\circ$). |
| `--bin_threshold`| Float | `0.15` | Target bin center radius [m] within which placement is counted as successful. |

---

## 5. Demonstration Recording & Policy Training Workflow

If retraining policies or fine-tuning from scratch:

```
              END-TO-END DATA GENERATION & TRAINING PIPELINE
              
    [Scripted State-Machine Controllers]
      • scripted_controller.py (Bolt)
      • coupler_scripted_controller.py (Coupler)
                       │
                       ▼
    [Demonstration Recording]
      • record_demos.py ──────────────► datasets/bin_transfer_expert_demos.h5
      • record_coupler_demos.py ──────► datasets/coupler_bin_transfer_expert_demos.h5
                       │
                       ├─────────────────────────────────┐
                       ▼                                 ▼
    [YOLOv8-Seg Fine-Tuning]            [Behavioral Cloning (BC) Training]
      • Extract RGB-D frames               • bc_pretrain.py
      • Train 25 epochs                    • 44-D Observation Vector
      • Output: yolov8n_bin_transfer_seg.pt• Output: bin_transfer_bc_policy.pt
                       │                                 │
                       └────────────────┬────────────────┘
                                        ▼
                        [Closed-Loop Evaluation]
                          • evaluate_with_perception.py
                          • evaluate_coupler_with_perception.py
```

### 5.1 Step 1: Record Expert Demonstrations
```bash
# Record 100 expert demonstrations for M20 Bolt
DISPLAY="" /home/azureuser/new_isaac/env_isaaclab/bin/python \
    scripts/reinforcement_learning/bin_transfer/record_demos.py \
    --num_envs 1 \
    --num_demos 100 \
    --output datasets/bin_transfer_expert_demos.h5

# Record 100 expert demonstrations for Bolt Coupler
DISPLAY="" /home/azureuser/new_isaac/env_isaaclab/bin/python \
    scripts/reinforcement_learning/bin_transfer/record_coupler_demos.py \
    --num_envs 1 \
    --num_demos 100 \
    --output datasets/coupler_bin_transfer_expert_demos.h5
```

### 5.2 Step 2: Behavioral Cloning Policy Training
```bash
# Train BC policy on recorded bolt demos
/home/azureuser/new_isaac/env_isaaclab/bin/python \
    scripts/reinforcement_learning/bin_transfer/bc_pretrain.py \
    --dataset datasets/bin_transfer_expert_demos.h5 \
    --output checkpoints/bin_transfer_bc_policy.pt \
    --epochs 50 \
    --batch_size 256 \
    --lr 3e-4
```

---

## 6. Code Quality, Formatting & Testing Guidelines

Before committing or pushing any modifications to the repository, adhere strictly to the IsaacLab contributor guidelines.

### 6.1 Pre-Commit Hooks
Run formatting and linting checks across all files:

```bash
# Run pre-commit checks on all files
./isaaclab.sh -f
```

Workflow:
1. Make code modifications.
2. Run `./isaaclab.sh -f` to format and validate.
3. Review auto-formatted changes.
4. Stage files with `git add` and commit.

### 6.2 Unit Testing
Run specific test suites using pytest:

```bash
# Test specific manipulation tests
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/
```

### 6.3 Style Constraints
- Follow **PEP 8** style rules.
- State **SI units** in docstrings for physical quantities: e.g., `[m]`, `[rad]`, `[m/s]`, `[N]`.
- Use modern Python type hints: `x | y`, `x | None` (avoid `typing.Optional`, `typing.Union`).
- New files must include the current year (2026) in the SPDX header:
  ```python
  # Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
  # All rights reserved.
  #
  # SPDX-License-Identifier: BSD-3-Clause
  ```

---

## 7. Troubleshooting & Common Failure Modes

### 7.1 Issue: `RuntimeError: CUDA error: an illegal memory access was encountered` on Checkpoint Load
- **Cause**: PyTorch 2.6+ loading tensors directly to GPU (`map_location="cuda:0"`) while Omniverse PhysX/RTX context is initializing.
- **Solution**: Load weights to CPU first with `weights_only=False`:
  ```python
  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  ```
  Then transfer the required sub-tensors (`state_mean`, `state_std`, `policy_state_dict`) to the device. (Already implemented in [`bin_transfer_policy.py`](file:///home/azureuser/new_isaac/IsaacLab/scripts/reinforcement_learning/bin_transfer/bin_transfer_policy.py#L31)).

### 7.2 Issue: `error: unrecognized arguments: --real_time`
- **Cause**: Script was launched with `--real_time` before the argument was added to `argparse`.
- **Solution**: The argument is now registered across both `evaluate_with_perception.py` and `evaluate_coupler_with_perception.py`. If adding to a new script, ensure:
  ```python
  parser.add_argument("--real_time", action="store_true", default=False, help="Pace simulation loop.")
  ```

### 7.3 Issue: Black / Blank Camera Frames on Step 0
- **Cause**: Synthetic sensors in Isaac Sim require one simulation step to initialize render pipelines.
- **Solution**: Both `BoltPerceptionModule` and `CouplerPerceptionModule` contain safety fallbacks: if `cam.data.output` is None or lacks `"rgb"`, they initialize at nominal coordinates $[0.55, 0.20, Z_{\text{nom}}]$ with `valid_mask = False`.

### 7.4 Issue: Headless VM Fails with `Cannot open display`
- **Cause**: Omniverse trying to connect to X11 when running over SSH or non-GUI sessions.
- **Solution**: Prefix the command with `DISPLAY=""` and add `--headless`:
  ```bash
  DISPLAY="" /home/azureuser/new_isaac/env_isaaclab/bin/python <script.py> --headless
  ```

### 7.5 Issue: Gripper Closes But Part Does Not Lift
- **Cause**: Grasp threshold mismatch between observation stage flags and actual gripper joint position.
- **Solution**: Ensure normalized gripper opening falls within the calibrated window:
  ```python
  clamping = (gripper_opening > 0.74) and (gripper_opening < 0.88)
  ```
  and clearance distance $d_{xy} < 45\,\text{mm}$, $EE_z < 0.13\,\text{m}$.
