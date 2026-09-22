"""Seeded histories from training interactions only."""
from collections import Counter, defaultdict
import random
import numpy as np

class TrainingHistories:
    """Original positive-conditioned TRAINING context without shared-list mutation.

    Only training labels are used. Re-sample each epoch from training neighbors;
    validation uses the independent, train-only query histories in the bundle.
    """
    def __init__(self, train, m):
        self.by_user, self.by_target = defaultdict(list), defaultdict(list)
        for u, _, j, _ in train:
            self.by_user[int(u)].append(int(j))
            self.by_target[int(j)].append(int(u))
        self.popular = [int(j) for j, _ in Counter(train[:, 2]).most_common()]
        self.popular_users = [int(u) for u, _ in Counter(train[:, 0]).most_common()]
        self.size, self.top_users = m['u_pb_num'], m['top_u']
        self.with_self = m['with_self_his']
        self.repeated = m['repeated_interact']

    def sample(self, rows, seed, epoch):
        rng = random.Random(f'{seed}:{epoch}:training_history')
        output = []
        for u, _, j, _ in rows:
            peers = list(self.by_target.get(int(j), self.popular_users[:self.top_users]))
            if not self.with_self:
                peers = [v for v in peers if v != int(u)]
            if len(peers) >= self.top_users:
                peers = rng.sample(peers, self.top_users)
            elif not peers:
                peers = [v for v in self.popular_users if self.with_self or v != int(u)][:self.top_users]
            history = [item for v in peers for item in self.by_user[v]]
            if self.repeated:
                if int(j) in history:
                    history.remove(int(j))
            else:
                history = [item for item in history if item != int(j)]
            history = rng.sample(history, min(self.size, len(history)))
            if len(history) < self.size:
                # Only the first few fallback items are needed, not the catalog.
                padding = []
                for item in self.popular:
                    if item != int(j):
                        padding.append(item)
                    if len(padding) >= self.size:
                        break
                if not padding:
                    raise ValueError('No non-target training item for history padding')
                while len(history) < self.size:
                    history.append(padding[len(history) % len(padding)])
            output.append(history)
        return np.asarray(output, dtype=np.int64)

def query_histories(train, rows, size, top_users, seed):
    """No validation/test labels or negative IDs influence history selection."""
    by_user, by_top = defaultdict(list), defaultdict(set)
    for u, i, j, _ in train:
        by_user[int(u)].append(int(j))
        by_top[int(i)].add(int(u))
    popular = [int(j) for j, _ in Counter(train[:, 2]).most_common()]
    cache = {}
    result = np.empty((len(rows), size), dtype=np.int64)
    for index, (u, i, *_unused) in enumerate(rows):
        key = (int(u), int(i))
        if key not in cache:
            rng = np.random.default_rng(np.random.SeedSequence([seed, *key]))
            users = np.array(sorted(by_top[int(i)] - {int(u)}), dtype=np.int64)
            if len(users) > top_users:
                users = rng.choice(users, top_users, replace=False)
            pool = [j for v in users for j in by_user[int(v)]]
            if not pool:
                pool = list(by_user[int(u)]) or popular
            chosen = list(rng.choice(pool, min(size, len(pool)), replace=False))
            chosen += [popular[t % len(popular)] for t in range(size - len(chosen))]
            cache[key] = chosen
        result[index] = cache[key]
    return result
