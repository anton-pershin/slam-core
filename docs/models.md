# Models

Models are abstractions that take an input and produce a prediction. They are used within the evaluation pipeline to generate responses on collections of evaluation cases.

## Core abstraction

### `Model`

An abstract base class for all models.

```python
from slam_core.model import Model

class MyModel(Model):
    def __init__(self, name: str):
        super().__init__(name)

    def predict(self, x: Any) -> Any:
        # Return a prediction given input x
        ...
```

---

## `LlmViaOpenAiApi`

A concrete model that sends prompts to an LLM server via the OpenAI-compatible API (powered by `rally`).

```python
from rally.llm import Llm
from slam_core.model import LlmViaOpenAiApi
from slam_core.collections.text_generation import TextGenerationInput

llm = Llm(
    url="http://localhost:8000",
    authorization="Bearer your-token",
    model="qwen3-8b",
    max_output_tokens=1024,
)

model = LlmViaOpenAiApi(name="qwen", llm=llm)

case = TextGenerationInput(
    system_prompt="You are a helpful assistant.",
    user_prompt="What is the capital of France?",
)

response = model.predict(case)
```

**How it works:**

1. If `system_prompt` is not `None`, it is added as a system message.
2. `user_prompt` is added as a user message.
3. The message list is sent to `rally`'s `request_based_on_message_history`.
4. Returns the `content` field of the assistant's response.

---

## `EmbeddingBasedTextClassifier`

A model that combines an embedding model with a text classifier. It embeds the input text, then passes the embedding through a classifier to produce a label.

```python
from slam_core.model import EmbeddingBasedTextClassifier

model = EmbeddingBasedTextClassifier(
    name="text_classifier",
    embedding_model=your_embedding_model,  # must implement predict(list[str]) -> embeddings
    classifier=your_classifier,            # must implement predict(embeddings) -> indices,
                                           # and expose .model_path and .labels
)

label = model.predict("This is a sample text")
```

The classifier must conform to `TextClassifierProtocol`:

- `model_path: str` — path to the classifier model.
- `labels: Sequence[str]` — list of possible label names.
- `predict(text_embeddings) -> Sequence` — returns predicted class indices.
