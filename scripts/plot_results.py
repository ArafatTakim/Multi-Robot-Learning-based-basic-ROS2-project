#!/usr/bin/env python3
"""Plot learning curves from one or more result CSVs.

Usage: python3 scripts/plot_results.py results/shared.csv results/independent.csv
"""
import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def load(path):
    by_ep = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_ep[int(row['episode'])].append(float(row['reward']))
    eps = sorted(by_ep)
    return eps, [sum(by_ep[e]) / len(by_ep[e]) for e in eps]


def smooth(values, window=15):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1):i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        eps, rewards = load(path)
        plt.plot(eps, smooth(rewards), label=path.split('/')[-1].replace('.csv', ''))
    plt.xlabel('Episode')
    plt.ylabel('Mean reward across robots (smoothed)')
    plt.title('Multi-robot Q-learning')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    print('Saved learning_curves.png')
    plt.show()
