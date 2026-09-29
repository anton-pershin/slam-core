"""Tests for slam_core.model.LocalCausalLm (KISS spec 03, T24-T30).

All tests use a tiny randomly-initialised causal LM built in-process from a small
transformers config - nothing is downloaded from the network, no GPU is needed.
"""

from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
pytest.importorskip("torch")

from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    LlamaConfig,
    PreTrainedTokenizerFast,
)

from slam_core.collections.text_generation import TextGenerationInput  # noqa: E402
from slam_core.model import LocalCausalLm  # noqa: E402

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<s>{{ message['role'] }}: {{ message['content'] }}"
    "{% endfor %}<s>assistant: "
)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """Build a tiny causal LM + chat-template tokenizer in-process."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    model_dir = tmp_path_factory.mktemp("tiny_model")

    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "hello": 3,
        "world": 4,
        "the": 5,
        "sky": 6,
        "is": 7,
        "blue": 8,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    bos, eos = tokenizer.token_to_id("<s>"), tokenizer.token_to_id("</s>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"<s>:0 $A:0 </s>:1",
        pair=f"$A:0 </s>:1 $B:1 </s>:1",
        special_tokens=[("<s>", bos), ("</s>", eos)],
    )

    tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<pad>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(model_dir)

    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(model_dir)

    return str(model_dir)


def _make_adapter(tiny_model_dir: str, adapter_dir: Path) -> None:
    from peft import LoraConfig, get_peft_model

    base = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    peft_model.save_pretrained(str(adapter_dir))


class TestLocalCausalLm:
    def test_predict_accepts_text_generation_input(self, tiny_model_dir):
        model = LocalCausalLm(
            name="tiny", base_model_path=tiny_model_dir, max_new_tokens=4
        )
        x = TextGenerationInput(system_prompt=None, user_prompt="hello world")
        result = model.predict(x)
        assert isinstance(result, str)

    @pytest.mark.parametrize("enable_thinking", [False, True])
    def test_thinking_setting_passed_to_chat_template(
        self, tiny_model_dir, enable_thinking, monkeypatch
    ):
        model = LocalCausalLm(
            name="tiny",
            base_model_path=tiny_model_dir,
            max_new_tokens=1,
            enable_thinking=enable_thinking,
        )
        calls = []
        original = model.tokenizer.apply_chat_template

        def record(messages, **kwargs):
            calls.append(kwargs)
            return original(messages, **kwargs)

        monkeypatch.setattr(model.tokenizer, "apply_chat_template", record)
        model.predict(TextGenerationInput(system_prompt=None, user_prompt="hello"))

        assert calls[0]["chat_template_kwargs"] == {"enable_thinking": enable_thinking}

    def test_unset_thinking_setting_not_passed_to_chat_template(
        self, tiny_model_dir, monkeypatch
    ):
        model = LocalCausalLm(
            name="tiny", base_model_path=tiny_model_dir, max_new_tokens=1
        )
        calls = []
        original = model.tokenizer.apply_chat_template

        def record(messages, **kwargs):
            calls.append(kwargs)
            return original(messages, **kwargs)

        monkeypatch.setattr(model.tokenizer, "apply_chat_template", record)
        model.predict(TextGenerationInput(system_prompt=None, user_prompt="hello"))

        assert "chat_template_kwargs" not in calls[0]

    @pytest.mark.parametrize("enable_thinking", [False, True, None])
    def test_thinking_markup_cleanup_modes(
        self, tiny_model_dir, enable_thinking, monkeypatch
    ):
        import torch

        model = LocalCausalLm(
            name="tiny",
            base_model_path=tiny_model_dir,
            max_new_tokens=1,
            enable_thinking=enable_thinking,
        )
        original_template = model.tokenizer.apply_chat_template
        prompt_text = original_template(
            [{"role": "user", "content": "hello"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_length = len(model.tokenizer(prompt_text)["input_ids"])
        monkeypatch.setattr(
            model.model,
            "generate",
            lambda **kwargs: torch.tensor(
                [model.tokenizer(prompt_text)["input_ids"] + [3]]
            ),
        )
        monkeypatch.setattr(
            model.tokenizer,
            "decode",
            lambda generated_ids, skip_special_tokens: (
                '<think>\n\n</think>\n\n{"name": "Alice"}'
            ),
        )

        result = model.predict(
            TextGenerationInput(system_prompt=None, user_prompt="hello")
        )

        if enable_thinking is False:
            assert result == '{"name": "Alice"}'
        else:
            assert result == '<think>\n\n</think>\n\n{"name": "Alice"}'
        assert prompt_length > 0

    def test_output_does_not_contain_prompt(self, tiny_model_dir):
        # The returned string must be exactly the decode of the tokens
        # generated beyond the prompt - i.e. the echoed prompt is stripped by
        # construction, not by a substring check a random continuation could
        # coincidentally violate.
        import torch

        model = LocalCausalLm(
            name="tiny", base_model_path=tiny_model_dir, max_new_tokens=4
        )
        x = TextGenerationInput(system_prompt=None, user_prompt="hello world")

        messages = [{"role": "user", "content": "hello world"}]
        prompt_text = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = model.tokenizer(prompt_text, return_tensors="pt").to(
            model.model.device
        )
        prompt_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = model.model.generate(
                **inputs, max_new_tokens=4, do_sample=False, temperature=None
            )
        expected = model.tokenizer.decode(
            output_ids[0][prompt_length:], skip_special_tokens=True
        )

        assert model.predict(x) == expected
        # And the prompt itself is not echoed back as a prefix.
        assert not model.predict(x).startswith(prompt_text)

    def test_greedy_decoding_is_deterministic(self, tiny_model_dir):
        model = LocalCausalLm(
            name="tiny", base_model_path=tiny_model_dir, max_new_tokens=4
        )
        x = TextGenerationInput(system_prompt=None, user_prompt="hello world")
        assert model.predict(x) == model.predict(x)

    def test_max_new_tokens_respected(self, tiny_model_dir):
        model = LocalCausalLm(
            name="tiny", base_model_path=tiny_model_dir, max_new_tokens=2
        )
        x = TextGenerationInput(system_prompt=None, user_prompt="hello world")
        result_ids = model.tokenizer(model.predict(x), add_special_tokens=False)[
            "input_ids"
        ]
        assert len(result_ids) <= 2

    def test_no_adapter_no_peft_wrapping(self, tiny_model_dir):
        from transformers import PreTrainedModel

        model = LocalCausalLm(
            name="tiny", base_model_path=tiny_model_dir, max_new_tokens=1
        )
        assert isinstance(model.model, PreTrainedModel)
        assert not hasattr(model.model, "peft_config")

    def test_with_adapter_produces_peft_model(self, tiny_model_dir, tmp_path):
        adapter_dir = tmp_path / "adapter"
        _make_adapter(tiny_model_dir, adapter_dir)

        model = LocalCausalLm(
            name="tiny",
            base_model_path=tiny_model_dir,
            adapter_path=str(adapter_dir),
            max_new_tokens=1,
        )
        assert isinstance(model.model, peft.PeftModel)

    def test_system_prompt_included_in_chat_template(self, tiny_model_dir):
        model = LocalCausalLm(
            name="tiny", base_model_path=tiny_model_dir, max_new_tokens=1
        )
        messages_with_system = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ]
        messages_without_system = [{"role": "user", "content": "hi"}]

        text_with = model.tokenizer.apply_chat_template(
            messages_with_system, tokenize=False
        )
        text_without = model.tokenizer.apply_chat_template(
            messages_without_system, tokenize=False
        )

        assert "be nice" in text_with
        assert "be nice" not in text_without
        assert "hi" in text_with
        assert "hi" in text_without


class TestStepCallback:
    def test_no_callback_unchanged(self, tiny_model_dir):
        x = TextGenerationInput(system_prompt=None, user_prompt="hello world")
        model_plain = LocalCausalLm(
            name="tiny", base_model_path=tiny_model_dir, max_new_tokens=4
        )
        model_none = LocalCausalLm(
            name="tiny",
            base_model_path=tiny_model_dir,
            max_new_tokens=4,
            step_callback=None,
        )
        assert model_plain.predict(x) == model_none.predict(x)

    def test_callback_invoked_per_step(self, tiny_model_dir):
        x = TextGenerationInput(system_prompt=None, user_prompt="hello world")
        calls = []
        model = LocalCausalLm(
            name="tiny",
            base_model_path=tiny_model_dir,
            max_new_tokens=4,
            step_callback=calls.append,
        )
        prediction = model.predict(x)
        assert isinstance(prediction, str)
        assert len(calls) >= 1  # tiny model generates at most 4 tokens
        assert all(isinstance(t, int) for t in calls)
