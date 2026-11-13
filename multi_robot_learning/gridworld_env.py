"""Environment node: simulates one grid world per robot.

Per robot i:
  sub  robot_i/action       std_msgs/Int32             (0=up 1=down 2=left 3=right)
  pub  robot_i/observation  std_msgs/Float32MultiArray [x, y, reward, done, step_id, success]

When an episode ends (done=1) the robot is reset and the observation carries the
new start position, so the learner can immediately act again.
"""
import time
from functools import partial

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32

from .core import GridWorld

START_POSITIONS = [(0, 0), (0, 7), (7, 0), (3, 0), (0, 4), (4, 7)]


class GridWorldEnv(Node):
    def __init__(self):
        super().__init__('gridworld_env')
        self.declare_parameter('num_robots', 3)
        self.declare_parameter('grid_size', 8)
        self.declare_parameter('max_steps', 100)

        n = self.get_parameter('num_robots').value
        size = self.get_parameter('grid_size').value
        max_steps = self.get_parameter('max_steps').value

        self.envs, self.pubs = [], []
        self.step_ids = [0] * n
        self.last_obs = [None] * n
        self.last_time = [0.0] * n

        for i in range(n):
            env = GridWorld(size=size, start=START_POSITIONS[i % len(START_POSITIONS)],
                            max_steps=max_steps)
            env.reset()
            self.envs.append(env)
            self.pubs.append(self.create_publisher(Float32MultiArray, f'robot_{i}/observation', 10))
            self.create_subscription(Int32, f'robot_{i}/action', partial(self.on_action, i), 10)
            self.publish_obs(i, env.pos, 0.0, False, False)

        # Re-send the latest observation if a robot goes quiet (e.g. it started late).
        self.create_timer(0.5, self.heartbeat)
        self.get_logger().info(f'GridWorld ready with {n} robots ({size}x{size}).')

    def publish_obs(self, i, pos, reward, done, success):
        msg = Float32MultiArray()
        msg.data = [float(pos[0]), float(pos[1]), float(reward), float(done),
                    float(self.step_ids[i]), float(success)]
        self.last_obs[i] = msg
        self.last_time[i] = time.monotonic()
        self.pubs[i].publish(msg)

    def on_action(self, i, msg):
        env = self.envs[i]
        pos, reward, done, success = env.step(msg.data)
        if done:
            pos = env.reset()
        self.step_ids[i] += 1
        self.publish_obs(i, pos, reward, done, success)

    def heartbeat(self):
        now = time.monotonic()
        for i, msg in enumerate(self.last_obs):
            if now - self.last_time[i] > 1.0:
                self.last_time[i] = now
                self.pubs[i].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GridWorldEnv()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
