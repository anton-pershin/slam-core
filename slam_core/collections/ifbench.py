from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import requests

from slam_core.collections.base import (
    CollectionInfo,
    EvalCaseCollection,
    check_if_loaded,
)
from slam_core.collections.text_generation import (
    TextGenerationInput,
    TextGenerationWithUniqueGroundTruth,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class _IFBenchExample:
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]


class IFBench(EvalCaseCollection):
    def __init__(self, name: str, jsonl_path: str, download_url: str) -> None:
        super().__init__(name)
        self.jsonl_path = Path(jsonl_path)
        self.download_url = download_url

    def _ensure_dataset(self) -> None:
        if self.jsonl_path.exists():
            return
        LOGGER.info("Downloading IFBench dataset to %s", self.jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(self.download_url, timeout=120)
        response.raise_for_status()
        self.jsonl_path.write_bytes(response.content)

    def _load(self) -> CollectionInfo:
        self._ensure_dataset()
        lines = self.jsonl_path.read_text(encoding="utf-8").splitlines()

        def _iterator() -> Iterator[_IFBenchExample]:
            for line in lines:
                if not line.strip():
                    continue
                payload = json.loads(line)
                yield _IFBenchExample(
                    prompt=payload["prompt"],
                    instruction_id_list=payload["instruction_id_list"],
                    kwargs=payload["kwargs"],
                )

        return CollectionInfo(collection=_iterator(), collection_len=len(lines))

    @check_if_loaded
    def __next__(self) -> TextGenerationWithUniqueGroundTruth:
        raw_item = next(self.collection)  # type: ignore
        metadata = {
            "instruction_id_list": raw_item.instruction_id_list,
            "kwargs": raw_item.kwargs,
        }
        return TextGenerationWithUniqueGroundTruth(  # type: ignore[misc]
            x=TextGenerationInput(
                system_prompt=None,
                user_prompt=raw_item.prompt,
            ),
            y_true=metadata,
        )
