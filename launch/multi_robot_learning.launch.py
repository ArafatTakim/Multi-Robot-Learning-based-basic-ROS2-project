from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'multi_robot_learning'


def _setup(context, *args, **kwargs):
    n = int(LaunchConfiguration('num_robots').perform(context))
    share = LaunchConfiguration('share').perform(context).lower() == 'true'
    episodes = int(LaunchConfiguration('max_episodes').perform(context))
    run_name = LaunchConfiguration('run_name').perform(context)

    nodes = [
        Node(package=PKG, executable='gridworld_env', output='screen',
             parameters=[{'num_robots': n}]),
        Node(package=PKG, executable='metrics_logger', output='screen',
             parameters=[{'num_robots': n, 'max_episodes': episodes, 'run_name': run_name}]),
    ]
    for i in range(n):
        nodes.append(Node(
            package=PKG, executable='q_learner', name=f'learner_{i}', output='screen',
            parameters=[{'robot_id': i, 'share_enabled': share, 'max_episodes': episodes}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('num_robots', default_value='3'),
        DeclareLaunchArgument('share', default_value='true',
                              description='Share Q-tables between robots'),
        DeclareLaunchArgument('max_episodes', default_value='300'),
        DeclareLaunchArgument('run_name', default_value='shared'),
        OpaqueFunction(function=_setup),
    ])
