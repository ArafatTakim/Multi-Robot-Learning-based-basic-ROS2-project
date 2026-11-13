from glob import glob

from setuptools import find_packages, setup

package_name = 'multi_robot_learning'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Basic multi-robot reinforcement learning with knowledge sharing in ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gridworld_env = multi_robot_learning.gridworld_env:main',
            'q_learner = multi_robot_learning.q_learner:main',
            'metrics_logger = multi_robot_learning.metrics_logger:main',
        ],
    },
)
