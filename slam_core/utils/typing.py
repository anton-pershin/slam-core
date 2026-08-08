from typing import Protocol


class HasStr(Protocol):
    def __str__(self) -> str: ...
