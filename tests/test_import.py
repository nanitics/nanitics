import pytest

import nanitics


def test_import_nanitics() -> None:
    assert nanitics is not None


@pytest.mark.parametrize("name", nanitics.__all__)
def test_all_exports_importable(name: str) -> None:
    """Every entry in __all__ must be an importable attribute."""
    assert hasattr(nanitics, name), f"{name} is listed in __all__ but not importable"


def test_import_context_management_types() -> None:
    from nanitics.context import (
        ContextManager,
        ContextUsage,
        EstimateTokenCounter,
        SummarizationPolicy,
        TokenCounter,
        TruncationPolicy,
    )
    from nanitics.infrastructure import (
        ContextAssemblyEvent,
        ContextSummarizationEvent,
        ContextTruncationEvent,
    )
    from nanitics.tracing import (
        ContextContribution,
        RemovedMessageInfo,
    )

    assert ContextManager is not None
    assert ContextUsage is not None
    assert EstimateTokenCounter is not None
    assert SummarizationPolicy is not None
    assert TokenCounter is not None
    assert TruncationPolicy is not None
    assert ContextTruncationEvent is not None
    assert ContextSummarizationEvent is not None
    assert ContextAssemblyEvent is not None
    assert ContextContribution is not None
    assert RemovedMessageInfo is not None


def test_import_embedding_types() -> None:
    from nanitics.infrastructure import (
        EmbeddingClient,
        MockEmbeddingClient,
    )

    assert EmbeddingClient is not None
    assert MockEmbeddingClient is not None


def test_import_semantic_memory_types() -> None:
    from nanitics.infrastructure import (
        SemanticDeleteEvent,
        SemanticSearchEvent,
        SemanticStoreEvent,
    )
    from nanitics.memory import (
        InMemorySemanticStore,
        SearchResult,
        SemanticStore,
        create_semantic_memory_tools,
    )

    assert SemanticStore is not None
    assert SearchResult is not None
    assert InMemorySemanticStore is not None
    assert create_semantic_memory_tools is not None
    assert SemanticStoreEvent is not None
    assert SemanticSearchEvent is not None
    assert SemanticDeleteEvent is not None
