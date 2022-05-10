import enum

from dataclasses import dataclass


class OpCode(enum.IntEnum):
    Init = 0
    Finalize = 1
    SetItem = 2
    Reshape = 3
    Permute = 4


@dataclass
class Selector:
    index: int


@dataclass
class CommunicationPrimitive:
    name: str


@dataclass
class Pipeline(CommunicationPrimitive):
    bidirectional: bool = True
    head_by_tail: bool = True


@dataclass
class Group(CommunicationPrimitive):
    pass
