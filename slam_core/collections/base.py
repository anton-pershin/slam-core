from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, Optional, TypedDict, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def check_if_loaded(func: F) -> F:
    def wrapper(self: EvalCaseCollection, *args: Any, **kwargs: Any):
        if self.collection is None:
            raise CollectionNotLoadedError(
                f"Collection {self.name} not loaded. "
                "Perhaps, you forgot to call load()"
            )
        return func(self, *args, **kwargs)

    return cast(F, wrapper)


class EvalCase(TypedDict):
    x: Any
    y_true: Any


class CollectionInfo(TypedDict):
    collection: Iterator[Any]
    collection_len: int


class EvalCaseCollection(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.collection: Optional[Iterator[Any]] = None
        self.collection_len: Optional[int] = None

    def __iter__(self) -> EvalCaseCollection:
        return self

    def load(self) -> None:
        collection_info = self._load()
        self.collection = collection_info["collection"]
        self.collection_len = collection_info["collection_len"]

    @check_if_loaded
    def __len__(self) -> int:
        if self.collection_len is None:
            raise CollectionNotLoadedError(
                f"Collection {self.name} not loaded. Perhaps, you forgot to call load()"
            )
        return self.collection_len

    @abstractmethod
    def _load(self) -> CollectionInfo: ...

    @abstractmethod
    def __next__(self) -> EvalCase: ...


class CollectionNotLoadedError(Exception):
    pass
