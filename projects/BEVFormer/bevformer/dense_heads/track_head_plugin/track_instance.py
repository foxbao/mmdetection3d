"""Track-instance container (vendored from UniAD with no behavioural change).

This is the dynamic-length query bag that the MOTR-style track loop carries
across frames. Each field is a tensor of length N; ``Instances[idx]`` slices
all fields together, so callers can do ``inst[scores > thr]`` without having
to remember which attributes exist.

We vendor it rather than adding a dependency on the UniAD checkout, but the
semantics (fields, len, cat, to, getitem) are identical to UniAD's copy.
"""
import itertools
from typing import Any, Dict, List, Tuple, Union

import torch


class Instances:
    """A list-like bag of per-track tensors with set-like field access."""

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any) -> None:
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith('_'):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == '_fields' or name not in self._fields:
            raise AttributeError(
                f"Cannot find field '{name}' in the given Instances!")
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        data_len = len(value)
        if len(self._fields):
            assert len(self) == data_len, (
                f'Adding a field of length {data_len} to an Instances of '
                f'length {len(self)}')
        self._fields[name] = value

    def has(self, name: str) -> bool:
        return name in self._fields

    def remove(self, name: str) -> None:
        del self._fields[name]

    def get(self, name: str) -> Any:
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        return self._fields

    def to(self, *args: Any, **kwargs: Any) -> 'Instances':
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def numpy(self) -> 'Instances':
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, 'numpy'):
                v = v.numpy()
            ret.set(k, v)
        return ret

    def __getitem__(
            self, item: Union[int, slice, torch.BoolTensor]) -> 'Instances':
        if type(item) is int:
            if item >= len(self) or item < -len(self):
                raise IndexError('Instances index out of range!')
            item = slice(item, None, len(self))
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            return v.__len__()
        raise NotImplementedError('Empty Instances does not support __len__!')

    def __iter__(self):
        raise NotImplementedError('`Instances` object is not iterable!')

    @staticmethod
    def cat(instance_lists: List['Instances']) -> 'Instances':
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), 'cat'):
                values = type(v0).cat(values)
            else:
                raise ValueError(
                    f'Unsupported type {type(v0)} for concatenation')
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + '('
        s += f'num_instances={len(self)}, '
        s += f'image_height={self._image_size[0]}, '
        s += f'image_width={self._image_size[1]}, '
        s += 'fields=[{}])'.format(', '.join(
            f'{k}: {v}' for k, v in self._fields.items()))
        return s

    __repr__ = __str__
