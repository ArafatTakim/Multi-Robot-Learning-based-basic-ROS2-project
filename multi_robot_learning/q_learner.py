"""Learner node: one Q-learning agent per robot, with optional knowledge sharing.

  sub  robot_<id>/observation  std_msgs/Float32MultiArray
  pub  robot_<id>/action       std_msgs/Int32
  pub  /episode_stats          [robot_id, episode, total_reward, steps, success, epsilon]
  pub  /q_table_share          [robot_id, <flattened Q-table>]
  sub  /q_table_share          (other robots' tables are merged into ours)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32

from .core import QAgent


class QLearner(Node):
    def __init__(self):
        super().__init__('q_learner')
        self.declare_parameter('robot_id', 0)
        self.declare_parameter('grid_size', 8)
        self.declare_parameter('alpha', 0.1)
        self.declare_parameter('gamma', 0.95)
        self.declare_parameter('epsilon', 1.0)
        self.declare_parameter('epsilon_min', 0.05)
        self.declare_parameter('epsilon_decay', 0.99)
        self.declare_parameter('max_episodes', 300)
        self.declare_parameter('share_enabled', True)
        self.declare_parameter('share_interval', 2.0)
        self.declare_parameter('share_weight', 0.5)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.robot_id = p('robot_id')
        self.max_episodes = p('max_episodes')
        self.share_enabled = p('share_enabled')
        self.share_weight = p('share_weight')

        self.agent = QAgent(size=p('grid_size'), alpha=p('alpha'), gamma=p('gamma'),
                            epsilon=p('epsilon'), epsilon_min=p('epsilon_min'),
                            epsilon_decay=p('epsilon_decay'), seed=self.robot_id)

        self.last_state = None
        self.last_action = None
        self.last_action_msg = None
        self.last_step_id = -1
        self.episode = 0
        self.ep_reward = 0.0
        self.ep_steps = 0
        self.finished = False

        rid = self.robot_id
        self.action_pub = self.create_publisher(Int32, f'robot_{rid}/action', 10)
        self.stats_pub = self.create_publisher(Float32MultiArray, '/episode_stats', 10)
        self.create_subscription(Float32MultiArray, f'robot_{rid}/observation', self.on_obs, 10)

        if self.share_enabled:
            self.share_pub = self.create_publisher(Float32MultiArray, '/q_table_share', 10)
            self.create_subscription(Float32MultiArray, '/q_table_share', self.on_share, 10)
            self.create_timer(p('share_interval'), self.broadcast_q)

        self.get_logger().info(f'Robot {rid} learner started (sharing={self.share_enabled}).')

    # ---- learning loop -------------------------------------------------
    def on_obs(self, msg):
        if self.finished:
            return
        x, y, reward, done, step_id, success = msg.data
        step_id = int(step_id)

        if step_id == self.last_step_id:          # env re-sent: our action was probably lost
            if self.last_action_msg is not None:
                self.action_pub.publish(self.last_action_msg)
            return
        if step_id < self.last_step_id:
            return
        self.last_step_id = step_id

        s2 = self.agent.state_index(x, y)
        if self.last_state is not None:
            self.agent.update(self.last_state, self.last_action, reward, s2, bool(done))
            self.ep_reward += reward
            self.ep_steps += 1

        if done:
            self.end_episode(bool(success))
            if self.finished:
                return

        a = self.agent.act(s2)
        self.last_state, self.last_action = s2, a
        self.last_action_msg = Int32(data=a)
        self.action_pub.publish(self.last_action_msg)

    def end_episode(self, success):
        self.episode += 1
        stats = Float32MultiArray()
        stats.data = [float(self.robot_id), float(self.episode), float(self.ep_reward),
                      float(self.ep_steps), float(success), float(self.agent.epsilon)]
        self.stats_pub.publish(stats)

        if self.episode % 10 == 0:
            self.get_logger().info(
                f'[robot {self.robot_id}] ep {self.episode:4d} | reward {self.ep_reward:7.1f} '
                f'| steps {self.ep_steps:3d} | success {success} | eps {self.agent.epsilon:.2f}')

        self.agent.decay()
        self.ep_reward, self.ep_steps = 0.0, 0
        self.last_state = None
        if self.episode >= self.max_episodes:
            self.finished = True
            self.get_logger().info(f'Robot {self.robot_id} finished {self.max_episodes} episodes.')

    # ---- knowledge sharing --------------------------------------------
    def broadcast_q(self):
        msg = Float32MultiArray()
        msg.data = [float(self.robot_id)] + self.agent.q.flatten().astype(float).tolist()
        self.share_pub.publish(msg)

    def on_share(self, msg):
        if int(msg.data[0]) == self.robot_id:
            return
        self.agent.merge(np.array(msg.data[1:]), self.share_weight)


def main(args=None):
    rclpy.init(args=args)
    node = QLearner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
