import numpy as np


def reshape(old_shape, *shape):
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
    volumn = np.prod(old_shape)

    if auto_shape_dim is not None:
        reshape_vol = -reshape_vol
        if volumn % reshape_vol != 0:
            raise ValueError
        else:
            _shape = list(shape)
            _shape[auto_shape_dim] = volumn // reshape_vol
    else:
        _shape = shape
        if reshape_vol != volumn:
            raise ValueError
    return _shape


def permute(old_shape, *dims):
    return tuple(np.array(old_shape)[dims].tolist())
