from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    Sequence,
    cast,
    runtime_checkable,
)

from kygs.classifier import TextClassifier
from rally.interaction import make_up_message_history
from rally.llm import Llm
from rally.thinking import THINKING_REMOVERS

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
        messages = make_up_message_history(
            system_prompt=x["system_prompt"],
            user_prompt=x["user_prompt"],
        )

        resp_message = self.llm.request(messages)

        if resp_message is None:
            raise ValueError(
                f"Model '{self.name}' received no valid response from the LLM."
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
        model_family: Optional[str] = None,
        remove_thinking: bool = False,
        step_callback: Optional[Callable[[int], None]] = None,
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
        self.model_family = model_family
        self.remove_thinking = remove_thinking
        self.step_callback = step_callback

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

    def _render_prompt(self, x: TextGenerationInput) -> str:
        """The exact prompt text the model is given for one input."""
        messages = make_up_message_history(
            system_prompt=x["system_prompt"],
            user_prompt=x["user_prompt"],
        )
        template_args: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.enable_thinking is not None:
            template_args["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }

        # A single conversation with tokenize=False renders to one string; the
        # template's annotation also allows its batched and tokenized forms.
        return cast(str, self.tokenizer.apply_chat_template(messages, **template_args))

    def prompt_token_count(self, x: TextGenerationInput) -> int:
        """How many tokens the prompt for this input takes."""
        return len(self.tokenizer(self._render_prompt(x))["input_ids"])

    def predict(self, x: TextGenerationInput) -> str:
        import torch

        prompt_text = self._render_prompt(x)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "temperature": self.temperature if self.do_sample else None,
            }
            if self.step_callback is not None:
                from transformers.generation import BaseStreamer

                class _CallbackStreamer(BaseStreamer):
                    """Forwards each decoding step to the user callback.

                    `put()` first receives the full prompt tensor (prefill echo),
                    then one tensor per generated token; only the latter are
                    forwarded, so the callback fires once per generated token.
                    """

                    def __init__(self, callback: Callable[[int], None]) -> None:
                        self._callback = callback
                        self._prompt_seen = False

                    def put(self, value: Any) -> None:
                        if self._prompt_seen:
                            self._callback(int(value.reshape(-1)[-1]))
                        else:
                            self._prompt_seen = True

                    def end(self) -> None:
                        pass

                generate_kwargs["streamer"] = _CallbackStreamer(self.step_callback)
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        generated_ids = output_ids[0][prompt_length:]
        completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if not isinstance(completion, str):
            completion = completion[0]
        if self.remove_thinking:
            completion = THINKING_REMOVERS[self.model_family](completion)

        return completion
