# multi_robot_learning

A minimal **ROS 2** project for **multi-robot reinforcement learning**. Several robots learn to
navigate a grid world to a shared goal using tabular Q-learning, and can optionally **share their
Q-tables** with each other over a ROS topic (a simple form of cooperative / federated learning).

It has no simulator dependency (no Gazebo), so it runs on any machine with ROS 2 and is a good
starting point to build on.

## Architecture

```
             robot_i/action (Int32)
  ┌─────────────┐ ───────────────────────▶ ┌──────────────────┐
  │ q_learner   │                           │ gridworld_env    │
  │ (robot i)   │ ◀─────────────────────── │ (one world/robot)│
  └─────┬───────┘  robot_i/observation      └──────────────────┘
        │ ▲        [x, y, reward, done, step_id, success]
        │ │
        ▼ │   /q_table_share  [robot_id, flattened Q-table]
   other learners  (merged into each robot's own Q-table)

  q_learner ──▶ /episode_stats ──▶ metrics_logger ──▶ results/<run>.csv
```

| Node | Role |
|------|------|
| `gridworld_env` | Simulates an independent 8x8 grid (with obstacles) per robot; goal at the top-right corner |
| `q_learner` | One per robot. Epsilon-greedy Q-learning; periodically broadcasts and merges Q-tables |
| `metrics_logger` | Records per-episode reward/steps/success to CSV |

Core logic (`GridWorld`, `QAgent`) lives in `multi_robot_learning/core.py` with no ROS imports,
so it is unit-tested without a ROS install.

## Requirements

- ROS 2 Humble, Jazzy, or newer
- `python3-numpy` (and `matplotlib` for plotting)

## Build

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/<your-username>/multi_robot_learning.git
cd ~/ros2_ws
rosdep install --from-paths src -y --ignore-src
colcon build --packages-select multi_robot_learning
source install/setup.bash
```

## Run

```bash
# 3 robots that share knowledge
ros2 launch multi_robot_learning multi_robot_learning.launch.py run_name:=shared

# 3 robots learning independently (for comparison)
ros2 launch multi_robot_learning multi_robot_learning.launch.py share:=false run_name:=independent
```

Launch arguments: `num_robots` (1-6, default 3), `share` (true/false), `max_episodes` (default 300),
`run_name` (CSV file name).

Plot and compare runs:

```bash
python3 scripts/plot_results.py results/shared.csv results/independent.csv
```

Run the unit tests:

```bash
python3 -m pytest test/
```

## Key parameters (`q_learner`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `alpha` | 0.1 | Learning rate |
| `gamma` | 0.95 | Discount factor |
| `epsilon` / `epsilon_min` / `epsilon_decay` | 1.0 / 0.05 / 0.99 | Exploration schedule |
| `share_enabled` | true | Broadcast and merge Q-tables |
| `share_interval` | 2.0 s | How often a robot broadcasts its Q-table |
| `share_weight` | 0.5 | Blend factor when merging another robot's values |

## How sharing works

Each robot publishes its Q-table to `/q_table_share`. When a robot receives another robot's
table, entries it has never visited (value 0) are copied over, and entries both have learned are
averaged using `share_weight`.

**Note:** on this small grid the benefit of sharing is modest, because each robot learns the map
quickly by itself. Sharing matters more as the environment grows or as robots start in different
regions. Try bigger grids, more robots, or different `share_weight` values and compare the curves.

## Ideas for extending

- Replace the grid world with Gazebo / TurtleBot3 and use real `/odom` and `/scan` data
- Swap Q-learning for DQN or PPO (e.g. with PyTorch) and share model weights
- Add robot-robot collisions and a shared map so robots must coordinate
- Use a custom `.msg` type instead of `Float32MultiArray`
- Add a decentralised variant where robots only share with neighbours in range

## License

MIT
