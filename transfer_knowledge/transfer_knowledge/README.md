# Transfer Knowledge: IsaacLab UR10e Bin-to-Bin Transfer

Welcome to the transfer knowledge documentation for the NVIDIA Isaac Lab UR10e Bin Transfer project. This directory consolidates the complete architectural specification, perception pipeline breakdown, and operational setup guide for incoming engineers and researchers.

---

## Documents in this Directory

1. **[Latest Codebase Architecture (`latest_codebase_architecture.md`)](file:///home/azureuser/new_isaac/IsaacLab/transfer_knowledge/latest_codebase_architecture.md)**
   - Complete technical breakdown of the **Look-Then-Move** perception-control paradigm.
   - Dual-Modality Segmentation: YOLOv8-Seg deep learning engine & GPU metric depth/cavity gating.
   - Pinhole optical backprojection and symmetry-aware 2nd central moment grasp yaw calculation.
   - 3-State Occlusion Latching State Machine (Active Tracking $\to$ Pose Latching $\to$ Proprioceptive FK Carry).
   - 44-D Observation Vector specification, 7-D Cartesian action space, and DLS Differential IK controller.
   - Repository structural map and post-mortem resolutions for critical runtime edge cases.

2. **[Setup & Operational Execution Guide (`setup_guide.md`)](file:///home/azureuser/new_isaac/IsaacLab/transfer_knowledge/setup_guide.md)**
   - System prerequisites, hardware/software stack, and Python 3.12 virtual environment setup.
   - Checklist of model weights and checkpoints (`bin_transfer_bc_policy.pt`, `coupler_bin_transfer_bc_policy.pt`, `yolov8n_bin_transfer_seg.pt`).
   - Copy-paste commands for headless evaluation and interactive Kit GUI visualization (`--real_time`).
   - End-to-end data collection (expert demos recording) and behavioral cloning training runbooks.
   - Code formatting, pre-commit checks (`./isaaclab.sh -f`), unit tests, and troubleshooting guide.

---

## Quick Reference Commands

### M20 Bolt Transfer Evaluation
```bash
# GUI Visualization
/home/azureuser/new_isaac/env_isaaclab/bin/python scripts/reinforcement_learning/bin_transfer/evaluate_with_perception.py --viz kit --real_time --num_envs 1 --num_episodes 5

# Headless Evaluation
DISPLAY="" /home/azureuser/new_isaac/env_isaaclab/bin/python scripts/reinforcement_learning/bin_transfer/evaluate_with_perception.py --headless --num_envs 1 --num_episodes 5
```

### Bolt Coupler Transfer Evaluation
```bash
# GUI Visualization
/home/azureuser/new_isaac/env_isaaclab/bin/python scripts/reinforcement_learning/bin_transfer/evaluate_coupler_with_perception.py --viz kit --real_time --num_envs 1 --num_episodes 5

# Headless Evaluation
DISPLAY="" /home/azureuser/new_isaac/env_isaaclab/bin/python scripts/reinforcement_learning/bin_transfer/evaluate_coupler_with_perception.py --headless --num_envs 1 --num_episodes 5
```
