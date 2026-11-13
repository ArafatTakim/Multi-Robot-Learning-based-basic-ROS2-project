import numpy as np

from multi_robot_learning.core import GOAL_REWARD, GridWorld, QAgent


def test_wall_bump_keeps_position():
    env = GridWorld(size=4, start=(0, 0), obstacles=set())
    pos, reward, done, _ = env.step(2)  # left, out of bounds
    assert pos == (0, 0) and reward < -1 and not done


def test_reaching_goal_ends_episode():
    env = GridWorld(size=2, start=(0, 0), goal=(1, 0), obstacles=set())
    pos, reward, done, success = env.step(3)  # right
    assert pos == (1, 0) and reward == GOAL_REWARD and done and success


def test_timeout_ends_episode():
    env = GridWorld(size=3, start=(0, 0), obstacles=set(), max_steps=2)
    env.step(2)
    _, _, done, success = env.step(2)
    assert done and not success


def test_q_update_moves_toward_target():
    agent = QAgent(size=2, alpha=0.5, gamma=0.9)
    agent.update(0, 1, 10.0, 1, done=True)
    assert agent.q[0, 1] == 5.0


def test_merge_copies_unvisited_and_blends_known():
    a, b = QAgent(size=2), QAgent(size=2)
    a.q[0, 0] = 4.0
    b.q[0, 0] = 8.0   # known by both -> blended
    b.q[1, 1] = 3.0   # known only by b -> copied
    a.merge(b.q.flatten(), weight=0.5)
    assert a.q[0, 0] == 6.0
    assert a.q[1, 1] == 3.0


def test_greedy_action_picks_best():
    agent = QAgent(size=2, epsilon=0.0, seed=0)
    agent.q[0] = np.array([0.0, 0.0, 5.0, 1.0])
    assert agent.act(0) == 2
