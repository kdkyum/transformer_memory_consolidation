import numpy as np
import networkx as nx


class RandomWalker:
    def __init__(self, L, num_envs, seed=0, n_x=10, n_a=5):
        self.num_envs = num_envs
        self.seed = seed
        self.L = L
        self.n_x = n_x
        self.n_a = n_a
        self.G = nx.grid_2d_graph(L, L)
        self.A = nx.to_numpy_array(self.G)
        if self.n_a == 5:
            self.A = np.identity(self.L ** 2) + self.A
        D = np.diag(np.sum(self.A, axis=0))
        self.T = np.linalg.inv(D) @ self.A
        self.L = L
        self._init_world()

    def _init_world(self):
        np.random.seed(self.seed)
        self.x = np.random.randint(0, self.n_x, size=(self.num_envs, self.L ** 2))
        self.valid_x = np.random.randint(0, self.n_x, size=(self.num_envs, self.L ** 2))

    def run(self, batch_size, seq_len, act_seed=1, validation=False, fix_start=False):
        np.random.seed(act_seed)
        obs = np.zeros((batch_size, seq_len + 1), dtype=np.int32)
        pos = np.zeros((batch_size, seq_len + 1), dtype=np.int32)
        act = np.zeros((batch_size, seq_len), dtype=np.int32)
        env_ids = np.random.choice(np.arange(self.num_envs), batch_size, replace=True)
        cum_T = np.cumsum(self.T, axis=1)
        if fix_start:
            init_pos = np.zeros((batch_size, 1), dtype=np.int32)
        else:
            init_pos = np.random.randint(0, self.L ** 2, size=(batch_size, 1))
        if validation:
            x = self.valid_x[env_ids]
        else:
            x = self.x[env_ids]
        obs[:, 0] = np.take_along_axis(x, init_pos, axis=1)[:, 0]
        pos[:, 0] = init_pos[:, 0]
        prev_pos = init_pos[:, 0]
        _rand = np.random.random((batch_size, seq_len, 1))
        for i in range(seq_len):
            _pos = (cum_T[prev_pos] < _rand[:, i]).sum(axis=1)
            pos[:, i + 1] = _pos
            obs[:, i + 1] = np.take_along_axis(x, _pos.reshape(batch_size, 1), axis=1)[
                :, 0
            ]
            tmp = _pos - prev_pos
            if self.n_a == 5:
                # a = 0 stay, a = 1 east, a = 2 south, a = 3 north, a = 4 west 
                tmp = np.where(tmp == self.L, 2, tmp)
                tmp = np.where(tmp == -self.L, 3, tmp)
                act[:, i] = np.where(tmp == -1, 4, tmp)
            else:
                tmp = np.where(tmp == self.L, 2, tmp)
                tmp = np.where(tmp == -self.L, 3, tmp)
                act[:, i] = np.where(tmp == -1, 0, tmp)
            prev_pos = _pos
        return pos, obs, act
