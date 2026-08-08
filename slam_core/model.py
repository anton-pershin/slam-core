from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, Sequence, runtime_checkable

from kygs.classifier import TextClassifier
from rally.interaction import request_based_on_message_history
from rally.llm import Llm

from slam_core.collections.text_generation import TextGenerationInput


class TextClassifierProtocol(Protocol):
    model_path: str
    labels: Sequence[str]

    def predict(self, text_embeddings: Any) -> Sequence[Any]: ...


@runtime_checkable
class _EmbeddingModel(Protocol):
    def predict(self, text_sequences: list[str]) -> Any: ...


class Model(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def predict(self, x: Any) -> Any: ...


class LlmViaOpenAiApi(Model):
    def __init__(self, name: str, llm: Llm) -> None:
        super().__init__(name)
        self.llm = llm

    def predict(self, x: TextGenerationInput) -> str:
        messages = []

        if x["system_prompt"] is not None:
            messages.append(
                {
                    "role": "system",
                    "content": x["system_prompt"],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": x["user_prompt"],
            }
        )

        resp_message = request_based_on_message_history(
            llm_server_url=self.llm.url,
            message_history=messages,
            authorization=self.llm.authorization,
            model=self.llm.model,
            max_output_tokens=self.llm.max_output_tokens,
        )

        return resp_message["content"]


class EmbeddingBasedTextClassifier(Model):
    def __init__(
        self,
        name: str,
        embedding_model: _EmbeddingModel,
        classifier: TextClassifier,
    ) -> None:
        super().__init__(name)
        self.embedding_model = embedding_model
        self.classifier: TextClassifierProtocol = classifier
        self.classifier_path = classifier.model_path

    def predict(self, x: str) -> str:
        embeddings = self.embedding_model.predict([x])
        predicted_indices = self.classifier.predict(embeddings)

        try:
            predicted_index_raw = predicted_indices[0]
        except (TypeError, IndexError) as err:  # pragma: no cover - defensive
            raise TypeError(
                "Classifier predict() must return an indexable sequence of predictions"
            ) from err

        try:
            predicted_index = int(predicted_index_raw)
        except (TypeError, ValueError) as err:  # pragma: no cover - defensive
            raise TypeError(
                "Classifier predict() must return an indexable sequence of predictions"
            ) from err

        try:
            label = self.classifier.labels[predicted_index]
        except (IndexError, TypeError) as err:  # pragma: no cover - defensive
            raise ValueError(
                f"Invalid class index {predicted_index} for labels "
                f"{self.classifier.labels}"
            ) from err

        return str(label)
