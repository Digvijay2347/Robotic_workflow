import os
from glob import glob
from setuptools import find_packages, setup

package_name = "ur3_rl_deployment"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "models"), glob("models/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Robotics Developer",
    maintainer_email="user@todo.todo",
    description="ROS 2 deployment package for trained IsaacLab RL reach policy on Universal Robots UR3",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "ur3_rl_policy_node = ur3_rl_deployment.ur3_rl_policy_node:main",
            "target_pose_publisher = ur3_rl_deployment.target_pose_publisher:main",
            "interactive_target_publisher = ur3_rl_deployment.interactive_target_publisher:main",
        ],
    },
)
