"""Meta AI OPT-175b setting; 992 NVIDIA A100-80GB GPUs"""

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
        from spmd_grid.comm import dp_sharded, mp_megatron

        n_layer = 100
        weight = 42
        grad = 22
        # forward
        for _ in range(n_layer):
            dp_sharded.allgather(weight)
            mp_megatron.allgather(weight)
            # ...
            mp_megatron.reduce_scatter(weight)

        # backward
        for _ in range(n_layer):
            dp_sharded.reduce_scatter(grad)
            mp_megatron.reduce_scatter(grad)
            # ...
            mp_megatron.allgather(grad)


# Physical layout: 124 nodes, 8 GPUs per node
def opt_175b(n_nodes=124, gpus_per_node=8):
    grid = Grid(Trainer, n_nodes, gpus_per_node)
    grid[:, X] = Group(name="mp_megatron")  # Megatron-LM model parallel
    grid[X, :] = Group(name="dp_sharded")  # fully sharded data parallel
    grid.set_options(num_gpus=1)
    return grid


if __name__ == "__main__":
    ray.init(num_gpus=48, num_cpus=48)
    grid = opt_175b(4, 8)
    actor_group = grid.remote()
    actor_group.wait_ready()  # optional
    actor_group.train_iter()

    # resize and automatic reshape
    print("=" * 40)
    new_actor_group = actor_group.resize(6, 8)
    new_actor_group.train_iter()
