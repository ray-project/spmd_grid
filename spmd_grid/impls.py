import numpy as np
from spmd_grid.primitives import OpCode, Selector, Group, Pipeline


class BidirectionalPipeline:
    def __init__(self):
        pass

    def forward_send(self, x):
        pass

    def forward_recv(self):
        pass

    def backward_send(self, x):
        pass

    def backward_recv(self):
        pass


class CollectiveCommunicationGroup:
    def broadcast(self, rank, data):
        pass

    def reduce(self, rank, data):
        pass

    def allreduce(self, data):
        pass

    def gather(self, rank, data):
        pass

    def scatter(self, rank, data):
        pass

    def reduce_scatter(self, data):
        pass

    def allgather(self, data):
        pass


def _reshape_pos(pos, old_shape, new_shape):
    ind = np.ravel_multi_index(pos, old_shape)
    return np.unravel_index(ind, new_shape)


def _init_spmd_comm(rank, logs):
    from spmd_grid import comm

    rlogs = list(reversed(logs))

    init_op, shape = rlogs.pop()
    assert init_op == OpCode.Init
    pos = np.unravel_index(rank, shape)

    while rlogs:
        op, op_data = rlogs.pop()
        if op == OpCode.Finalize:
            break
        elif op == OpCode.Reshape:
            pos = _reshape_pos(pos, old_shape=shape, new_shape=op_data)
            shape = op_data
        elif op == OpCode.Permute:
            pos = np.array(pos)[op_data].tolist()
        elif op == OpCode.SetItem:
            key, value = op_data
            # TODO(suquark): Implement initialization of collective communications
            if isinstance(value, Group):
                setattr(comm, value.name, CollectiveCommunicationGroup())
            elif isinstance(value, Pipeline):
                setattr(comm, value.name, BidirectionalPipeline())
            else:
                assert False
        else:
            assert False

    assert not rlogs, rlogs

    comm.shape = tuple(shape)
    comm.rank = tuple(pos)
