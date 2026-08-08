from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, TypedDict

import datasets

from slam_core.collections.base import (
    CollectionInfo,
    EvalCaseCollection,
    check_if_loaded,
)


class TextGenerationInput(TypedDict):
    system_prompt: Optional[str]
    user_prompt: str


class TextGenerationWithUniqueGroundTruth(TypedDict):
    x: TextGenerationInput
    y_true: str | dict[str, Any]


class TextGenerationMetadata(TypedDict, total=False):
    metadata: dict[str, Any]


class BigBenchHard(EvalCaseCollection):
    def __init__(
        self,
        name: str,
        dataset_name: str,
        split: str,
        subset: str,
        user_prompt_template: str,
    ) -> None:
        super().__init__(name)
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset
        self.user_prompt_template = user_prompt_template

    def _load(self) -> CollectionInfo:
        collection = datasets.load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
        )
        return CollectionInfo(
            collection=iter(collection),
            collection_len=collection.num_rows,
        )

    @check_if_loaded
    def __next__(self) -> TextGenerationWithUniqueGroundTruth:
        raw_item = next(self.collection)  # type: ignore
        return TextGenerationWithUniqueGroundTruth(  # type: ignore[misc]
            x=TextGenerationInput(
                system_prompt=None,
                user_prompt=self.user_prompt_template.format(
                    original_input=raw_item["input"],
                ),
            ),
            y_true=raw_item["target"],
        )


@dataclass
class _MergeQualityExample:
    attributes: dict[str, Any]
    unique_identifiers: dict[str, Any]
    chunks: list[str]


class MergeQuality(EvalCaseCollection):
    def __init__(
        self,
        name: str,
        jsonl_path: str,
        user_prompt_template: str,
    ) -> None:
        super().__init__(name)
        self.jsonl_path = Path(jsonl_path).expanduser()
        self.user_prompt_template = user_prompt_template

    def _load(self) -> CollectionInfo:
        raw_lines = self.jsonl_path.read_text(encoding="utf-8").splitlines()
        non_empty_lines = [line for line in raw_lines if line.strip()]

        def _iterator() -> Iterator[_MergeQualityExample]:
            for line in non_empty_lines:
                payload = json.loads(line)
                ground_truth = payload["ground_truth"]
                attributes = ground_truth["attributes"]
                unique_identifiers = payload["provided_identifiers"]
                chunks = [chunk["content"] for chunk in payload.get("chunks", [])]

                yield _MergeQualityExample(
                    attributes=attributes,
                    unique_identifiers=unique_identifiers,
                    chunks=chunks,
                )

        return CollectionInfo(
            collection=_iterator(), collection_len=len(non_empty_lines)
        )

    @staticmethod
    def _format_unique_identifiers(unique_identifiers: dict[str, Any]) -> str:
        return json.dumps(unique_identifiers, ensure_ascii=False, indent=2)

    @staticmethod
    def _format_chunks(chunks: list[str]) -> str:
        return "\n\n".join(chunks)

    @check_if_loaded
    def __next__(self) -> TextGenerationWithUniqueGroundTruth:
        raw_item = next(self.collection)  # type: ignore
        if isinstance(raw_item, dict):  # pragma: no cover - defensive
            attributes = raw_item["attributes"]
            unique_identifiers = raw_item["unique_identifiers"]
            chunks = raw_item["chunks"]
        else:
            attributes = raw_item.attributes
            unique_identifiers = raw_item.unique_identifiers
            chunks = raw_item.chunks

        user_prompt = self.user_prompt_template.format(
            unique_identifiers=self._format_unique_identifiers(unique_identifiers),
            data_chunks=self._format_chunks(chunks),
        )

        return TextGenerationWithUniqueGroundTruth(  # type: ignore[misc]
            x=TextGenerationInput(
                system_prompt=None,
                user_prompt=user_prompt,
            ),
            y_true=attributes,
        )
