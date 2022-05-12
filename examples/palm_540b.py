"""Google pathways language model 540b; 6144 TPUv4"""

import ray

from spmd_grid.grid import Grid
from spmd_grid.primitives import Selector as X
from spmd_grid.primitives import Group

from spmd_grid import comm


class Trainer:
    def __init__(self):
        self.rank = comm.rank
        self.shape = comm.shape
        print(f"world_shape={self.shape}, rank={self.rank}")

    def train_iter(self):
        from spmd_grid.comm import data_parallel_replica, data_parallel_sharded, model_parallel_sharded

        n_layer = 100
        weight = 42
        grad = 22
        # forward
        for _ in range(n_layer):
            model_parallel_sharded.allgather(weight)
            data_parallel_sharded.allgather(weight)
            # ...
            model_parallel_sharded.reduce_scatter(weight)

        # backward
        for _ in range(n_layer):
            model_parallel_sharded.reduce_scatter(grad)
            data_parallel_sharded.reduce_scatter(grad)
            # ...
            model_parallel_sharded.allgather(grad)

        data_parallel_replica.allreduce(grad)


# Physical layout: 2 nodes, 3072 TPUv4-32GB per node
def palm_540b(n_nodes=2, tpus_per_node=3072):
    # Physical layout: 2 nodes, 3072 TPUs per node
    grid = Grid(Trainer, n_nodes, tpus_per_node)
    # 12-way sharded model parallel, 256-way sharded data parallel
    grid.reshape(2, -1, 12)
    grid[X, :, :] = Group(name="data_parallel_replica")
    grid[:, X, :] = Group(name="data_parallel_sharded")
    grid[:, :, X] = Group(name="model_parallel_sharded")
    return grid


if __name__ == "__main__":
    ray.init(num_gpus=48, num_cpus=48)

    grid = palm_540b(2, 24)
    grid_handle = grid.remote()
    grid_handle.wait_ready()  # optional
    grid_handle.train_iter()

    # resize and automatic reshape
    print("=" * 40)
    grid_handle.resize(1, 24)
    grid_handle.train_iter()
