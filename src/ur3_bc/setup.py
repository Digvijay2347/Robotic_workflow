import os
from glob import glob
from setuptools import find_packages, setup

package_name = "ur3_bc"

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
    maintainer="surender",
    maintainer_email="surender@todo.todo",
    description="ROS 2 deployment package for UR3 Behavioral Cloning policy (Task 1 & Task 2)",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "task1_reach_target = ur3_bc.task1_reach_target:main",
            "task2_pick_and_place = ur3_bc.task2_pick_and_place:main",
            "send_target = ur3_bc.send_target:main",
        ],
    },
)
