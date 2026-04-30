from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nanitics.infrastructure.embeddings import EmbeddingClient, MockEmbeddingClient, VoyageEmbeddingClient
from nanitics.infrastructure.errors import EmbeddingProviderError, EmbeddingRateLimitError


class TestMockEmbeddingClient:
    async def test_returns_vectors_of_configured_dimension(self) -> None:
        client = MockEmbeddingClient(dimension=128)
        vectors = await client.embed(["hello"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 128

    async def test_default_dimension(self) -> None:
        client = MockEmbeddingClient()
        vectors = await client.embed(["hello"])
        assert len(vectors[0]) == 1024

    async def test_batch_input_returns_matching_output(self) -> None:
        client = MockEmbeddingClient(dimension=64)
        texts = ["one", "two", "three"]
        vectors = await client.embed(texts)
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == 64

    async def test_deterministic_same_text(self) -> None:
        client = MockEmbeddingClient(dimension=64)
        v1 = (await client.embed(["hello"]))[0]
        v2 = (await client.embed(["hello"]))[0]
        assert v1 == v2

    async def test_different_texts_yield_different_vectors(self) -> None:
        client = MockEmbeddingClient(dimension=64)
        vectors = await client.embed(["hello", "world"])
        assert vectors[0] != vectors[1]

    async def test_calls_tracked(self) -> None:
        client = MockEmbeddingClient(dimension=16)
        await client.embed(["a", "b"])
        await client.embed(["c"])
        assert len(client.calls) == 2
        assert client.calls[0] == ["a", "b"]
        assert client.calls[1] == ["c"]

    async def test_vectors_are_unit_normalized(self) -> None:
        client = MockEmbeddingClient(dimension=128)
        vectors = await client.embed(["test"])
        magnitude = sum(x * x for x in vectors[0]) ** 0.5
        assert abs(magnitude - 1.0) < 1e-6

    async def test_empty_input_returns_empty(self) -> None:
        client = MockEmbeddingClient(dimension=64)
        vectors = await client.embed([])
        assert vectors == []

    def test_satisfies_protocol(self) -> None:
        client = MockEmbeddingClient()
        assert isinstance(client, EmbeddingClient)


class TestVoyageEmbeddingClientUnit:
    def _make_client(self) -> VoyageEmbeddingClient:
        from nanitics.infrastructure.embeddings import VoyageEmbeddingClient

        return VoyageEmbeddingClient(api_key="test-key")

    def test_missing_api_key_raises_value_error(self) -> None:
        from nanitics.infrastructure.embeddings import VoyageEmbeddingClient

        with patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError, match="Voyage API key required"):
            VoyageEmbeddingClient()

    async def test_rate_limit_error_mapping(self) -> None:
        client = self._make_client()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "30"}

        mock_request = MagicMock(spec=httpx.Request)
        exc = httpx.HTTPStatusError("rate limited", request=mock_request, response=mock_response)

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = exc
            with pytest.raises(EmbeddingRateLimitError) as exc_info:
                await client.embed(["hello"])
            assert exc_info.value.retry_after == 30.0

    async def test_rate_limit_without_retry_after(self) -> None:
        client = self._make_client()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {}

        mock_request = MagicMock(spec=httpx.Request)
        exc = httpx.HTTPStatusError("rate limited", request=mock_request, response=mock_response)

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = exc
            with pytest.raises(EmbeddingRateLimitError) as exc_info:
                await client.embed(["hello"])
            assert exc_info.value.retry_after is None

    async def test_server_error_mapping(self) -> None:
        client = self._make_client()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.headers = {}

        mock_request = MagicMock(spec=httpx.Request)
        exc = httpx.HTTPStatusError("server error", request=mock_request, response=mock_response)

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = exc
            with pytest.raises(EmbeddingProviderError) as exc_info:
                await client.embed(["hello"])
            assert exc_info.value.status_code == 500
            assert exc_info.value.provider == "voyage"

    async def test_auth_error_mapping(self) -> None:
        client = self._make_client()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.headers = {}

        mock_request = MagicMock(spec=httpx.Request)
        exc = httpx.HTTPStatusError("unauthorized", request=mock_request, response=mock_response)

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = exc
            with pytest.raises(EmbeddingProviderError) as exc_info:
                await client.embed(["hello"])
            assert exc_info.value.status_code == 401
            assert exc_info.value.provider == "voyage"

    async def test_connection_error_mapping(self) -> None:
        client = self._make_client()
        mock_request = MagicMock(spec=httpx.Request)
        exc = httpx.ConnectError("connection failed", request=mock_request)

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = exc
            with pytest.raises(EmbeddingProviderError) as exc_info:
                await client.embed(["hello"])
            assert exc_info.value.provider == "voyage"
            assert exc_info.value.status_code is None

    def test_satisfies_protocol(self) -> None:
        client = self._make_client()
        assert isinstance(client, EmbeddingClient)

    async def test_successful_embed(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.embed(["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    async def test_close(self) -> None:
        client = self._make_client()
        with patch.object(client._client, "aclose", new_callable=AsyncMock) as mock_close:
            await client.close()
        mock_close.assert_awaited_once()
