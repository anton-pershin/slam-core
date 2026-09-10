from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from slam_core.model import EmbeddingBasedTextClassifier


class DummyClassifier:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.model_path = "dummy_path"
        self._predictions: list[int] = []
        self.captured_embeddings: list[Any] = []
        self.call_count = 0

    def predict(self, embeddings):
        self.captured_embeddings.append(embeddings)
        result = np.array([self._predictions[self.call_count]])
        self.call_count += 1
        return result


class TestEmbeddingBasedTextClassifier:
    def test_predict_returns_expected_labels(self, monkeypatch):
        fake_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        fake_texts = ["sample 1", "sample 2"]
        labels = ["class_a", "class_b", "class_c"]

        mock_embedding_model = Mock()
        mock_embedding_model.predict.side_effect = [
            fake_embeddings[0:1],
            fake_embeddings[1:2],
        ]

        dummy_classifier = DummyClassifier(labels)
        dummy_classifier._predictions = [1, 2]

        classifier = EmbeddingBasedTextClassifier(
            name="test_classifier",
            embedding_model=mock_embedding_model,
            classifier=dummy_classifier,
        )

        predicted_labels = [classifier.predict(text) for text in fake_texts]

        assert predicted_labels == ["class_b", "class_c"]
        assert mock_embedding_model.predict.call_count == 2
        assert dummy_classifier.captured_embeddings[0].shape == (1, 2)

    def test_predict_raises_when_classifier_returns_scalar(self, monkeypatch):
        mock_embedding_model = Mock()
        mock_embedding_model.predict.return_value = np.array([[0.1, 0.2]])

        dummy_classifier = DummyClassifier(["a"])
        dummy_classifier._predictions = []

        classifier = EmbeddingBasedTextClassifier(
            name="test_classifier",
            embedding_model=mock_embedding_model,
            classifier=dummy_classifier,
        )

        dummy_classifier.predict = lambda _: 1  # type: ignore[assignment]

        with pytest.raises(TypeError):
            classifier.predict("sample")
