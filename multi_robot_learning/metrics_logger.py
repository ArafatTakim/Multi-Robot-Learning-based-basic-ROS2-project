"""Logs /episode_stats from all robots to a CSV file for later plotting."""
import csv
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class MetricsLogger(Node):
    def __init__(self):
        super().__init__('metrics_logger')
        self.declare_parameter('output_dir', 'results')
        self.declare_parameter('run_name', 'run')
        self.declare_parameter('num_robots', 3)
        self.declare_parameter('max_episodes', 300)

        out_dir = self.get_parameter('output_dir').value
        run = self.get_parameter('run_name').value
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f'{run}.csv')
        self.file = open(self.path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['robot_id', 'episode', 'reward', 'steps', 'success', 'epsilon'])

        self.expected = self.get_parameter('num_robots').value
        self.max_episodes = self.get_parameter('max_episodes').value
        self.done_robots = set()

        self.create_subscription(Float32MultiArray, '/episode_stats', self.on_stats, 100)
        self.get_logger().info(f'Writing metrics to {self.path}')

    def on_stats(self, msg):
        rid, ep, reward, steps, success, eps = msg.data
        self.writer.writerow([int(rid), int(ep), f'{reward:.2f}', int(steps), int(success), f'{eps:.3f}'])
        self.file.flush()
        if int(ep) >= self.max_episodes:
            self.done_robots.add(int(rid))
            if len(self.done_robots) >= self.expected:
                self.get_logger().info(f'All robots finished. Results saved to {self.path} (Ctrl+C to exit).')

    def destroy_node(self):
        self.file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MetricsLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
