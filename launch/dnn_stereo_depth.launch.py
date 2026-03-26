# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import launch

from ament_index_python.packages import get_package_share_directory

from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for testing relevant nodes."""
    launch_args = [
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='["/workspaces/data/20-26-39/feature_runner_fp16.engine", "/workspaces/data/20-26-39/post_runner_fp16.engine"]',
            description='The absolute file path to the TensorRT engine file'),
        DeclareLaunchArgument(
            'model_type',
            default_value='FAST_FOUNDATION_STEREO',
            choices=['FAST_FOUNDATION_STEREO'],
            description='Model type'),
        DeclareLaunchArgument(
            'input_image_width',
            default_value='640',
            description='The input image width'),
        DeclareLaunchArgument(
            'input_image_height',
            default_value='480',
            description='The input image height'),
        DeclareLaunchArgument(
            'model_input_width',
            default_value='640',
            description='The model input width'),
        DeclareLaunchArgument(
            'model_input_height',
            default_value='480',
            description='The model input height'),
        DeclareLaunchArgument(
            'trigger_on_demands',
            default_value='false',
            description='trigger_on_demands'),
    ]

    # Image preprocessing parameters
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    model_input_width = LaunchConfiguration('model_input_width')
    model_input_height = LaunchConfiguration('model_input_height')

    model_type = LaunchConfiguration('model_type')

    trigger_on_demands = LaunchConfiguration('trigger_on_demands')

    # TensorRT parameters
    engine_file_path = LaunchConfiguration('engine_file_path')

    dnn_stereo_depth_node = Node(
        package="ros_dnn_stereo",
        executable="dnn_stereo_depth_node",
        name="dnn_stereo_depth_node",
        parameters=[
            {"image_reliability": 1},
            {"cam_info_reliability": 1},

            {"model_type": model_type},

            {"model_input_height": model_input_height},
            {"model_input_width": model_input_width},
            {"input_image_height": input_image_height},
            {"input_image_width": input_image_width},

            {"engine_file_path": engine_file_path},

            {"trigger_on_demands": trigger_on_demands},
        ],
        remappings=[
            ("left_ir_image", "/camera_head/left_ir/image_raw"),
            ("right_ir_image", "/camera_head/right_ir/image_raw"),

            ("disparity", "/camera_head/disparity/image_raw"),
        ],
    )

    disparity2pc_node = Node(
        package="ros_dnn_stereo",
        executable="disparity2pc_node",
        name="disparity2pc_node",
        parameters=[
            {"baseline": 0.095},
            {"zfar": 10.0},
            {"zmin": 0.1},
        ],
        remappings=[
            ("disparity", "/camera_head/disparity/image_raw"),
            ("camera_info", "/camera_head/left_ir/camera_info"),

            ('points', '/camera_head/depth_gen/points'),
            ("depth_image", "/camera_head/depth_gen/image_raw"),
        ],
    )

    # Ros2 bag
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '/workspaces/data/head_cam/rosbag2_2026_02_25-11_44_49', '--start-offset 0 -r 1 -l'],
        shell=True, output='screen')   


    rviz_config_path = os.path.join(get_package_share_directory(
        'ros_dnn_stereo'), 'rviz', 'dnn_stereo_depth.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen')


    final_launch_description = launch_args + [dnn_stereo_depth_node, disparity2pc_node, bag_play, rviz_node]

    # final_launch_description = launch_args + [dnn_stereo_depth_node, disparity2depth_node, bag_play, rviz_node]
    # final_launch_description = launch_args + [dnn_stereo_depth_node, disparity2depth_node]
    return launch.LaunchDescription(final_launch_description)
