"""Config composition tests (KISS spec 03, T33)."""

import pytest

hydra = pytest.importorskip("hydra")
transformers = pytest.importorskip("transformers")

from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from slam_core.model import LocalCausalLm  # noqa: E402
from tests.test_local_causal_lm import CHAT_TEMPLATE  # noqa: E402


def _build_tiny_model(root):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

    model_dir = root / "tiny"
    model_dir.mkdir()
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
    tok_raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tok_raw.pre_tokenizer = pre_tokenizers.Whitespace()
    bos, eos = tok_raw.token_to_id("<s>"), tok_raw.token_to_id("</s>")
    tok_raw.post_processor = processors.TemplateProcessing(
        single="<s>:0 $A:0 </s>:1",
        pair="$A:0 </s>:1 $B:1 </s>:1",
        special_tokens=[("<s>", bos), ("</s>", eos)],
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_raw,
        unk_token="<pad>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(model_dir)

    config = LlamaConfig(
        vocab_size=9,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(model_dir)
    return str(model_dir)


class TestLocalHfCausalLmConfig:
    def test_config_targets_local_causal_lm(self):
        cfg = OmegaConf.load("config/model/local_hf_causal_lm.yaml")
        assert cfg._target_ == "slam_core.model.LocalCausalLm"
        assert cfg.do_sample is False

    def test_instantiate_binds_generation_params(self, tmp_path):
        model_path = _build_tiny_model(tmp_path)
        cfg = OmegaConf.load("config/model/local_hf_causal_lm.yaml")
        cfg = OmegaConf.merge(cfg, {"base_model_path": model_path, "max_new_tokens": 4})
        model = instantiate(cfg)

        assert isinstance(model, LocalCausalLm)
        assert model.name == "local_hf_causal_lm"
        assert model.max_new_tokens == 4
        assert model.do_sample is False
        assert model.temperature == 1.0
        assert model.adapter_path is None
