from __future__ import annotations

from typing import Iterator, TypedDict

from kygs.message_provider import Message, MessageProvider

from slam_core.collections.base import (
    CollectionInfo,
    EvalCaseCollection,
    check_if_loaded,
)


class TextClassificationWithUniqueGroundTruth(TypedDict):
    x: str
    y_true: str


class RedditPostsSmallDataset(EvalCaseCollection):
    def __init__(self, name: str, message_provider: MessageProvider) -> None:
        super().__init__(name)
        self.message_provider = message_provider

    def _load(self) -> CollectionInfo:
        messages = self.message_provider.messages

        def _iter_messages() -> Iterator[Message]:
            for message in messages:
                yield message

        return CollectionInfo(collection=_iter_messages(), collection_len=len(messages))

    @check_if_loaded
    def __next__(self) -> TextClassificationWithUniqueGroundTruth:
        assert self.collection is not None  # check_if_loaded ensures this
        message = next(self.collection)
        label = message.true_label if message.true_label is not None else message.label
        if label is None:
            raise ValueError("Message does not contain either true_label or label")

        return TextClassificationWithUniqueGroundTruth(
            x=message.text,
            y_true=str(label),
        )
