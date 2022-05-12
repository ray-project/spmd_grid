"""3D parallelism"""

import ray

from spmd_grid.grid import Grid
from spmd_grid.primitives import Selector as X
from spmd_grid.primitives import Group, Pipeline

from spmd_grid import comm


class Trainer:
    def __init__(self):
        self.rank = comm.rank
        self.shape = comm.shape
        print(f"world_shape={self.shape}, rank={self.rank}")

    def hello(self):
        print("hello")


if __name__ == '__main__':
    ray.init(num_cpus=32)

    # 3D parallel
    # Physical layout: 6 nodes, 4 GPUs per node
    grid = Grid(Trainer, 6, 4)
    grid.reshape(2, -1, 4)
    grid[X, :, :] = Group(name="data_parallel")
    grid[:, X(0), X(1)] = Pipeline(name="pipe")
    grid[:, :, X] = Group(name="model_parallel")

    grid_handle = grid.remote()
    grid_handle.wait_ready()  # optional
    grid_handle.hello()

    print("=" * 40)
    grid_handle.resize(8, 4)
    grid_handle.hello()

