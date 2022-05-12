import numpy as np

import ray
from spmd_grid.impls import _init_spmd_comm
from spmd_grid.primitives import CommunicationPrimitive, OpCode
from spmd_grid.utils import reshape, permute


class GridHandle:
    def __init__(self, actor_cls, actor_options, logs, *args, **kwargs):
        self._actor_cls = actor_cls
        self._actor_options = actor_options
        self._logs = logs.copy()
        self._args = args
        self._kwargs = kwargs
        self._actors = []
        self._remote()

    def _remote(self):
        class SPMDActor(self._actor_cls):
            def __init__(self, rank, logs, *args, **kwargs):
                _init_spmd_comm(rank, logs)
                super().__init__(*args, **kwargs)

            def spmd_init_finished(self) -> bool:
                """This method is just for checking if the
                remote actor has been initialized."""
                return True

        if self._actor_options:
            ray_actor = ray.remote(**self._actor_options)(SPMDActor)
        else:
            ray_actor = ray.remote(SPMDActor)

        _, shape = self._logs[0]

        self._actors = []
        for r in range(np.prod(shape)):
            self._actors.append(ray_actor.remote(r, self._logs, *self._args, **self._kwargs))

    def wait_ready(self):
        ray.get([actor.spmd_init_finished.remote() for actor in self._actors])

    def resize(self, *new_shape):
        for a in self._actors:
            ray.kill(a)
        self._actors = []
        self._logs[0] = (OpCode.Init, tuple(new_shape))
        self._remote()

    def __getattr__(self, item):
        def _invoke(*args, **kwargs):
            ray.get(
                [getattr(actor, item).remote(*args, **kwargs) for actor in self._actors]
            )

        return _invoke


class Grid:
    def __init__(self, actor_cls, *shape):
        self._shape = shape
        self._actor_cls = actor_cls
        self._actor_options = {}
        self._logs = [(OpCode.Init, tuple(self._shape))]

    @property
    def shape(self):
        return tuple(self._shape)

    def reshape(self, *shape):
        self._shape = reshape(self._shape, *shape)
        self._logs.append((OpCode.Reshape, tuple(shape)))

    def permute(self, *dims):
        if list(sorted(dims)) != list(range(len(self._shape))):
            raise ValueError
        self._shape = permute(self._shape)
        self._logs.append((OpCode.Permute, tuple(dims)))

    def __setitem__(self, key, value):
        if not isinstance(value, CommunicationPrimitive):
            raise TypeError
        if isinstance(key, tuple):
            self._logs.append((OpCode.SetItem, (key, value)))
        else:
            raise NotImplementedError

    def set_options(self, **actor_options):
        self._actor_options = actor_options

    def remote(self, *args, **kwargs):
        self._logs.append((OpCode.Finalize, self._shape))
        return GridHandle(self._actor_cls, self._actor_options, self._logs, *args, **kwargs)
