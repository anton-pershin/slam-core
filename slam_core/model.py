from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, runtime_checkable

from kygs.classifier import TextClassifier
from rally.interaction import request_based_on_message_history
from rally.llm import Llm

from slam_core.collections.text_generation import TextGenerationInput

if TYPE_CHECKING:  # pragma: no cover - imports are heavy, kept lazy at runtime
    from peft import PeftModel
    from transformers import PreTrainedTokenizerBase


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


class LocalCausalLm(Model):
    """Local HuggingFace causal LM, optionally wrapped with a LoRA adapter.

    Heavy libraries (``torch``, ``transformers``, ``peft``) are imported lazily
    inside ``__init__`` so that importing this module stays cheap.
    """

    def __init__(
        self,
        name: str,
        base_model_path: str,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        device: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ) -> None:
        super().__init__(name)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model: Any

        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.device = device
        self.enable_thinking = enable_thinking

        torch_dtype = torch.float32 if device in (None, "cpu") else torch.float16
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
        )
        if adapter_path is None:
            model = base_model
        else:
            model = PeftModel.from_pretrained(base_model, adapter_path)
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            base_model_path
        )
        model.to(device if device is not None else "cpu")
        model.eval()
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, x: TextGenerationInput) -> str:
        import torch

        messages: list[dict[str, str]] = []
        if x["system_prompt"] is not None:
            messages.append({"role": "system", "content": x["system_prompt"]})
        messages.append({"role": "user", "content": x["user_prompt"]})

        chat_template_kwargs = (
            {"enable_thinking": self.enable_thinking}
            if self.enable_thinking is not None
            else None
        )
        template_args: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if chat_template_kwargs is not None:
            template_args["chat_template_kwargs"] = chat_template_kwargs
        prompt_text = self.tokenizer.apply_chat_template(messages, **template_args)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
            )

        generated_ids = output_ids[0][prompt_length:]
        completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if self.enable_thinking is False:
            completion = re.sub(
                r"<think>.*?</think>\s*", "", completion, flags=re.DOTALL
            ).strip()
        return completion if isinstance(completion, str) else completion[0]
