# Using a local LLM

Any OpenAI-compatible server works with `OpenAILLMClient` — set `base_url` to the server's URL and pass whatever API key the server expects. No dedicated local-LLM client is needed.

## Ollama

Ollama exposes an OpenAI-compatible endpoint at `http://localhost:11434/v1`. Pull a model with `ollama pull llama3.2` before running.

```python
from nanitics.infrastructure import OpenAILLMClient

llm = OpenAILLMClient(
    model="llama3.2",
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # any non-empty string; Ollama does not check keys
)
```

## vLLM

vLLM exposes an OpenAI-compatible server via `python -m vllm.entrypoints.openai.api_server --model <hf-id>`. The default port is `8000`.

```python
from nanitics.infrastructure import OpenAILLMClient

llm = OpenAILLMClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)
```

## LM Studio

LM Studio's Local Server tab exposes an OpenAI-compatible endpoint at `http://localhost:1234/v1`. Start the server from the app before running.

```python
from nanitics.infrastructure import OpenAILLMClient

llm = OpenAILLMClient(
    model="llama-3.2-3b-instruct",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)
```

## If you already use LiteLLM

If `nanitics[litellm]` is installed, `LiteLLMClient(model="ollama/llama3")` reaches the same endpoints through LiteLLM's routing layer. Prefer `OpenAILLMClient` when the endpoint is OpenAI-compatible — native error classification is stronger and the dependency footprint is lighter.

## See also

- `OpenAILLMClient` docstring in [`nanitics/infrastructure/llm/openai.py`](../../nanitics/infrastructure/llm/openai.py) — full constructor signature and streaming/tool-use semantics.
- [Getting Started](./getting-started.md) — first-agent walkthrough.
