import numpy as np

import ray
from spmd_grid.impls import _init_spmd_comm
from spmd_grid.primitives import CommunicationPrimitive, OpCode


class ActorGroup:
    def __init__(self, grid, actors, *args, **kwargs):
        self.grid = grid
        self.actors = actors
        self.args = args
        self.kwargs = kwargs

    def wait_ready(self):
        ray.get([actor.spmd_init_finished.remote() for actor in self.actors])

    def resize(self, *new_shape):
        for a in self.actors:
            ray.kill(a)
        self.grid._logs[0] = (OpCode.Init, tuple(new_shape))
        self.grid._logs.pop()
        return self.grid.remote(*self.args, **self.kwargs)

    def __getattr__(self, item):
        def _invoke(*args, **kwargs):
            ray.get(
                [getattr(actor, item).remote(*args, **kwargs) for actor in self.actors]
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
        auto_shape_dim = None
        for i, s in enumerate(shape):
            if not isinstance(s, int) or s == 0 or s < -1:
                raise ValueError(shape)
            if s == -1:
                if auto_shape_dim is not None:
                    raise ValueError(shape)
                else:
                    auto_shape_dim = i

        reshape_vol = np.prod(shape)
        volumn = np.prod(self._shape)

        if auto_shape_dim is not None:
            reshape_vol = -reshape_vol
            if volumn % reshape_vol != 0:
                raise ValueError
            else:
                shape = list(shape)
                shape[auto_shape_dim] = volumn // reshape_vol
        elif reshape_vol != volumn:
            raise ValueError
        self._logs.append((OpCode.Reshape, tuple(shape)))
        self._shape = shape

    def permute(self, *dims):
        if list(sorted(dims)) != list(range(len(self._shape))):
            raise ValueError
        self._logs.append((OpCode.Permute, tuple(dims)))
        self._shape = np.array(self._shape)[dims].tolist()

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

        actors = []
        for r in range(np.prod(self._shape)):
            actors.append(ray_actor.remote(r, self._logs, *args, **kwargs))
        return ActorGroup(self, actors, *args, **kwargs)
