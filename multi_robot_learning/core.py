"""Pure-Python core logic (no ROS imports) so it can be unit-tested anywhere."""
import numpy as np

# (dx, dy) for: up, down, left, right
ACTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]
ACTION_NAMES = ['up', 'down', 'left', 'right']

DEFAULT_OBSTACLES = frozenset(
    {(2, 1), (2, 2), (2, 3), (5, 4), (5, 5), (5, 6), (3, 5), (4, 2)}
)

STEP_PENALTY = -1.0
WALL_PENALTY = -5.0   # added on top of the step penalty when bumping into a wall/obstacle
GOAL_REWARD = 50.0


class GridWorld:
    """Single-robot grid world. Coordinates are (x, y)."""

    def __init__(self, size=8, start=(0, 0), goal=None,
                 obstacles=DEFAULT_OBSTACLES, max_steps=100):
        self.size = size
        self.start = start
        self.goal = goal if goal is not None else (size - 1, size - 1)
        self.obstacles = {o for o in obstacles
                          if 0 <= o[0] < size and 0 <= o[1] < size}
        self.max_steps = max_steps
        self.pos = start
        self.steps = 0

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self.pos

    def step(self, action):
        """Apply an action. Returns (pos, reward, done, success)."""
        dx, dy = ACTIONS[int(action)]
        x, y = self.pos
        nx, ny = x + dx, y + dy
        reward = STEP_PENALTY
        if not (0 <= nx < self.size and 0 <= ny < self.size) or (nx, ny) in self.obstacles:
            nx, ny = x, y
            reward += WALL_PENALTY
        self.pos = (nx, ny)
        self.steps += 1
        success = self.pos == self.goal
        if success:
            reward = GOAL_REWARD
        done = success or self.steps >= self.max_steps
        return self.pos, reward, done, success


class QAgent:
    """Tabular epsilon-greedy Q-learning agent."""

    def __init__(self, size=8, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.99, seed=None):
        self.size = size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((size * size, len(ACTIONS)))

    def state_index(self, x, y):
        return int(y) * self.size + int(x)

    def act(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(ACTIONS)))
        row = self.q[state]
        return int(self.rng.choice(np.flatnonzero(row == row.max())))

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * np.max(self.q[s2])
        self.q[s, a] += self.alpha * (target - self.q[s, a])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def merge(self, other, weight=0.5):
        """Blend another robot's Q-table into this one.

        Entries that are exactly 0 are treated as 'never visited': they are
        copied over instead of averaged, so unexplored cells don't dilute
        what the other robot has already learned.
        """
        other = np.asarray(other, dtype=float).reshape(self.q.shape)
        known = other != 0
        fresh = known & (self.q == 0)
        blend = known & ~fresh
        self.q[blend] = (1 - weight) * self.q[blend] + weight * other[blend]
        self.q[fresh] = other[fresh]
