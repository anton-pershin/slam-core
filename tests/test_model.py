"""Tests for slam_core.model."""

import inspect
import json
from unittest.mock import Mock, patch

import pytest
from rally.llm import Llm

import slam_core.model as model_module
from slam_core.collections.text_generation import TextGenerationInput
from slam_core.model import LlmViaOpenAiApi


def make_llm() -> Mock:
    """A Llm double carrying the configuration the real object would hold."""
    llm = Mock(spec=Llm)
    llm.url = "http://test-url.com"
    llm.authorization = "Bearer test-token"
    llm.model = "test-model"
    llm.max_output_tokens = 1000
    return llm


class TestLlmViaOpenAiApi:
    def test_init(self):
        mock_llm = make_llm()

        model = LlmViaOpenAiApi("test_model", mock_llm)

        assert model.name == "test_model"
        assert model.llm == mock_llm

    def test_predict_passes_system_and_user_messages(self):
        mock_llm = make_llm()
        mock_llm.request.return_value = {
            "role": "assistant",
            "content": "Test response",
        }

        model = LlmViaOpenAiApi("test_model", mock_llm)
        input_data = TextGenerationInput(
            system_prompt="You are a helpful assistant",
            user_prompt="Hello, world!",
        )
        model.predict(input_data)

        mock_llm.request.assert_called_once_with(
            [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello, world!"},
            ]
        )

    def test_predict_passes_user_message_only(self):
        mock_llm = make_llm()
        mock_llm.request.return_value = {
            "role": "assistant",
            "content": "Test response",
        }

        model = LlmViaOpenAiApi("test_model", mock_llm)
        input_data = TextGenerationInput(
            system_prompt=None,
            user_prompt="Hello, world!",
        )
        model.predict(input_data)

        mock_llm.request.assert_called_once_with(
            [{"role": "user", "content": "Hello, world!"}]
        )

    def test_predict_returns_llm_content(self):
        mock_llm = make_llm()
        mock_llm.request.return_value = {
            "role": "assistant",
            "content": "This is the model's response",
        }

        model = LlmViaOpenAiApi("test_model", mock_llm)
        input_data = TextGenerationInput(
            system_prompt=None,
            user_prompt="What is 2+2?",
        )
        result = model.predict(input_data)

        assert result == "This is the model's response"

    def test_predict_raises_when_llm_returns_none(self):
        mock_llm = make_llm()
        mock_llm.request.return_value = None

        model = LlmViaOpenAiApi("test_model", mock_llm)
        input_data = TextGenerationInput(
            system_prompt=None,
            user_prompt="What is 2+2?",
        )

        with pytest.raises(ValueError, match="test_model"):
            model.predict(input_data)

    def test_predict_passes_no_generation_parameters(self):
        mock_llm = make_llm()
        mock_llm.request.return_value = {
            "role": "assistant",
            "content": "Response content",
        }

        model = LlmViaOpenAiApi("test_model", mock_llm)
        input_data = TextGenerationInput(
            system_prompt=None,
            user_prompt="What is the capital of France?",
        )
        model.predict(input_data)

        mock_llm.request.assert_called_once()
        assert mock_llm.request.call_args.kwargs == {}
        assert mock_llm.request.call_args.args == (
            [{"role": "user", "content": "What is the capital of France?"}],
        )

    def test_predict_issues_one_request_per_call(self):
        mock_llm = make_llm()
        mock_llm.request.return_value = {
            "role": "assistant",
            "content": "Test response",
        }

        model = LlmViaOpenAiApi("test_model", mock_llm)
        model.predict(TextGenerationInput(system_prompt=None, user_prompt="First"))
        model.predict(TextGenerationInput(system_prompt=None, user_prompt="Second"))

        assert mock_llm.request.call_count == 2
        assert [call.args[0] for call in mock_llm.request.call_args_list] == [
            [{"role": "user", "content": "First"}],
            [{"role": "user", "content": "Second"}],
        ]

    def test_predict_returns_reasoning_trace_verbatim(self):
        content = "<think>2+2 is 4</think>4"
        mock_llm = make_llm()
        mock_llm.request.return_value = {"role": "assistant", "content": content}

        model = LlmViaOpenAiApi("test_model", mock_llm)
        input_data = TextGenerationInput(
            system_prompt=None,
            user_prompt="What is 2+2?",
        )

        assert model.predict(input_data) == content

    def test_module_surface_has_no_removed_request_functions(self):
        assert not hasattr(model_module, "request_based_on_message_history")
        assert not hasattr(model_module, "request_based_on_prompts")
        assert model_module.Llm is Llm

        source = inspect.getsource(model_module)
        assert "rally.interaction" not in source
        assert "request_based_on" not in source

    @patch("rally.llm.requests.post")
    def test_predict_sends_thinking_key_through_llm(self, mock_post):
        response = Mock()
        response.text = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
        mock_post.return_value = response

        def predict_with(**llm_kwargs) -> dict:
            llm = Llm(
                url="http://test-url.com",
                max_concurrent_requests=1,
                model="test-model",
                max_output_tokens=1000,
                **llm_kwargs,
            )
            model = LlmViaOpenAiApi("test_model", llm)
            result = model.predict(
                TextGenerationInput(system_prompt=None, user_prompt="Hello")
            )
            assert result == "ok"
            return json.loads(mock_post.call_args.kwargs["data"])

        with_thinking = predict_with(enable_thinking=True)
        assert with_thinking["chat_template_kwargs"] == {"enable_thinking": True}

        without_thinking = predict_with(enable_thinking=False)
        assert without_thinking["chat_template_kwargs"] == {"enable_thinking": False}

        unset = predict_with()
        assert "chat_template_kwargs" not in unset
        assert set(unset) == set(with_thinking) - {"chat_template_kwargs"}
        assert unset["model"] == "test-model"
        assert unset["messages"] == [{"role": "user", "content": "Hello"}]
