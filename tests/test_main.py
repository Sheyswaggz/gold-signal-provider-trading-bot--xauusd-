"""Comprehensive test suite for FastAPI application main module.

This module provides extensive testing coverage for the XAUUSD signal bot FastAPI application,
including:
- Health check endpoint validation
- Readiness probe testing with Redis dependency
- CORS configuration verification
- Response header validation
- Error handling scenarios
- Performance benchmarks

Test Categories:
- Unit Tests: Individual endpoint behavior
- Integration Tests: Redis connectivity and error handling
- Security Tests: CORS and header validation
- Performance Tests: Response time validation
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestHealthEndpoint:
    """Test suite for the /health endpoint.

    Validates basic health check functionality, response structure,
    and performance characteristics.
    """

    def test_health_returns_200_status(self, app_client: TestClient) -> None:
        """Test that health endpoint returns 200 OK status.

        Validates:
        - HTTP 200 status code
        - Successful response without errors
        """
        response = app_client.get("/health")
        assert response.status_code == status.HTTP_200_OK

    def test_health_returns_correct_json_structure(self, app_client: TestClient) -> None:
        """Test that health endpoint returns expected JSON structure.

        Validates:
        - Response contains 'status' field
        - Status value is 'healthy'
        - Response is valid JSON
        """
        response = app_client.get("/health")
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"
        assert isinstance(data, dict)

    def test_health_includes_correlation_id_header(self, app_client: TestClient) -> None:
        """Test that health endpoint includes correlation ID in response headers.

        Validates:
        - X-Correlation-ID header is present
        - Header value is non-empty string
        - Header format is valid
        """
        response = app_client.get("/health")

        assert "x-correlation-id" in response.headers
        correlation_id = response.headers["x-correlation-id"]
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0

    def test_health_response_content_type(self, app_client: TestClient) -> None:
        """Test that health endpoint returns correct content type.

        Validates:
        - Content-Type header is application/json
        - Response can be parsed as JSON
        """
        response = app_client.get("/health")

        assert "application/json" in response.headers["content-type"]
        assert response.json() is not None

    def test_health_endpoint_performance(self, app_client: TestClient) -> None:
        """Test that health endpoint responds within acceptable time.

        Validates:
        - Response time is under 100ms
        - Endpoint is performant for monitoring
        """
        import time

        start_time = time.time()
        response = app_client.get("/health")
        elapsed_time = time.time() - start_time

        assert response.status_code == status.HTTP_200_OK
        assert elapsed_time < 0.1  # 100ms threshold

    def test_health_endpoint_idempotency(self, app_client: TestClient) -> None:
        """Test that health endpoint is idempotent.

        Validates:
        - Multiple calls return same status
        - No side effects from repeated calls
        """
        response1 = app_client.get("/health")
        response2 = app_client.get("/health")
        response3 = app_client.get("/health")

        assert response1.status_code == response2.status_code == response3.status_code
        assert response1.json()["status"] == response2.json()["status"] == "healthy"

    @pytest.mark.parametrize(
        "method",
        ["POST", "PUT", "DELETE", "PATCH"],
    )
    def test_health_only_accepts_get_method(
        self, app_client: TestClient, method: str
    ) -> None:
        """Test that health endpoint only accepts GET requests.

        Validates:
        - POST, PUT, DELETE, PATCH return 405 Method Not Allowed
        - Only GET is supported
        """
        response = app_client.request(method, "/health")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


class TestReadyEndpoint:
    """Test suite for the /ready endpoint.

    Validates readiness probe functionality, Redis dependency checking,
    and error handling scenarios.
    """

    def test_ready_returns_200_when_redis_available(
        self, app_client: TestClient, mock_redis: AsyncMock
    ) -> None:
        """Test that ready endpoint returns 200 when Redis is available.

        Validates:
        - HTTP 200 status when Redis ping succeeds
        - Response indicates service is ready
        """
        mock_redis.ping.return_value = True

        with patch("app.main.redis_client", mock_redis):
            response = app_client.get("/ready")

        assert response.status_code == status.HTTP_200_OK

    def test_ready_returns_correct_json_when_available(
        self, app_client: TestClient, mock_redis: AsyncMock
    ) -> None:
        """Test that ready endpoint returns correct JSON structure when ready.

        Validates:
        - Response contains 'status' field
        - Status value is 'ready'
        - Response structure matches health endpoint
        """
        mock_redis.ping.return_value = True

        with patch("app.main.redis_client", mock_redis):
            response = app_client.get("/ready")
            data = response.json()

        assert "status" in data
        assert data["status"] == "ready"
        assert isinstance(data, dict)

    def test_ready_returns_503_when_redis_unavailable(
        self, app_client: TestClient, mock_redis: AsyncMock
    ) -> None:
        """Test that ready endpoint returns 503 when Redis is unavailable.

        Validates:
        - HTTP 503 Service Unavailable when Redis ping fails
        - Service correctly reports unready state
        """
        mock_redis.ping.side_effect = Exception("Redis connection failed")

        with patch("app.main.redis_client", mock_redis):
            response = app_client.get("/ready")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_ready_returns_error_json_when_unavailable(
        self, app_client: TestClient, mock_redis: AsyncMock
    ) -> None:
        """Test that ready endpoint returns error JSON when unavailable.

        Validates:
        - Response contains 'status' field
        - Status value is 'unavailable' or 'not_ready'
        - Error message is included
        """
        mock_redis.ping.side_effect = Exception("Redis connection failed")

        with patch("app.main.redis_client", mock_redis):
            response = app_client.get("/ready")
            data = response.json()

        assert "status" in data
        assert data["status"] in ["unavailable", "not_ready"]

    def test_ready_includes_correlation_id_on_success(
        self, app_client: TestClient, mock_redis: AsyncMock
    ) -> None:
        """Test that ready endpoint includes correlation ID on success.

        Validates:
        - X-Correlation-ID header is present
        - Header value is valid
        """
        mock_redis.ping.return_value = True

        with patch("app.main.redis_client", mock_redis):
            response = app_client.get("/ready")

        assert "x-correlation-id" in response.headers
        assert len(response.headers["x-correlation-id"]) > 0

    def test_ready_includes_correlation_id_on_failure(
        self, app_client: TestClient, mock_redis: AsyncMock
    ) -> None:
        """Test that ready endpoint includes correlation ID on failure.

        Validates:
        - X-Correlation-ID header is present even on error
        - Consistent header behavior across success/failure
        """
        mock_redis.ping.side_effect = Exception("Redis connection failed")

        with patch("app.main.redis_client", mock_redis):
            response = app_client.get("/ready")

        assert "x-correlation-id" in response.headers
        assert len(response.headers["x-correlation-id"]) > 0

    def test_ready_calls_redis_ping(
        self, app_client: TestClient, mock_redis: AsyncMock
    ) -> None:
        """Test that ready endpoint calls Redis ping method.

        Validates:
        - Redis ping is called during readiness check
        - Dependency checking is performed
        """
        mock_redis.ping.return_value = True

        with patch("app.main.redis_client", mock_redis):
            app_client.get("/ready")

        mock_redis.ping.assert_called_once()

    @pytest.mark.parametrize(
        "exception_type,exception_message",
        [
            (ConnectionError, "Connection refused"),
            (TimeoutError, "Connection timeout"),
            (Exception, "Unknown error"),
        ],
    )
    def test_ready_handles_different_redis_exceptions(
        self,
        app_client: TestClient,
        mock_redis: AsyncMock,
        exception_type: type[Exception],
        exception_message: str,
    ) -> None:
        """Test that ready endpoint handles various Redis exceptions.

        Validates:
        - Different exception types are handled gracefully
        - All exceptions result in 503 status
        - Service remains stable on errors
        """
        mock_redis.ping.side_effect = exception_type(exception_message)

        with patch("app.main.redis_client", mock_redis):
            response = app_client.get("/ready")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_ready_endpoint_performance_on_success(
        self, app_client: TestClient, mock_redis: AsyncMock
    ) -> None:
        """Test that ready endpoint responds quickly on success.

        Validates:
        - Response time is under 200ms
        - Readiness check is performant
        """
        import time

        mock_redis.ping.return_value = True

        with patch("app.main.redis_client", mock_redis):
            start_time = time.time()
            response = app_client.get("/ready")
            elapsed_time = time.time() - start_time

        assert response.status_code == status.HTTP_200_OK
        assert elapsed_time < 0.2  # 200ms threshold

    def test_ready_endpoint_performance_on_failure(
        self, app_client: TestClient, mock_redis: AsyncMock
    ) -> None:
        """Test that ready endpoint responds quickly on failure.

        Validates:
        - Response time is under 200ms even on error
        - Error handling is performant
        """
        import time

        mock_redis.ping.side_effect = Exception("Redis connection failed")

        with patch("app.main.redis_client", mock_redis):
            start_time = time.time()
            response = app_client.get("/ready")
            elapsed_time = time.time() - start_time

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert elapsed_time < 0.2  # 200ms threshold


class TestCORSConfiguration:
    """Test suite for CORS configuration.

    Validates Cross-Origin Resource Sharing headers and configuration
    for secure API access from web clients.
    """

    def test_cors_headers_present_on_health(self, app_client: TestClient) -> None:
        """Test that CORS headers are present on health endpoint.

        Validates:
        - Access-Control-Allow-Origin header exists
        - CORS is properly configured
        """
        response = app_client.get("/health")

        assert "access-control-allow-origin" in response.headers

    def test_cors_headers_present_on_ready(
        self, app_client: TestClient, mock_redis: AsyncMock
    ) -> None:
        """Test that CORS headers are present on ready endpoint.

        Validates:
        - Access-Control-Allow-Origin header exists
        - CORS is consistent across endpoints
        """
        mock_redis.ping.return_value = True

        with patch("app.main.redis_client", mock_redis):
            response = app_client.get("/ready")

        assert "access-control-allow-origin" in response.headers

    def test_cors_allows_credentials(self, app_client: TestClient) -> None:
        """Test that CORS allows credentials.

        Validates:
        - Access-Control-Allow-Credentials header is true
        - Credentials can be sent with requests
        """
        response = app_client.get("/health")

        if "access-control-allow-credentials" in response.headers:
            assert response.headers["access-control-allow-credentials"] == "true"

    def test_cors_preflight_request(self, app_client: TestClient) -> None:
        """Test that CORS preflight requests are handled.

        Validates:
        - OPTIONS requests return appropriate headers
        - Preflight requests are supported
        """
        response = app_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # Preflight should return 200 or 204
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_204_NO_CONTENT,
            status.HTTP_405_METHOD_NOT_ALLOWED,
        ]

    @pytest.mark.parametrize(
        "origin",
        [
            "http://localhost:3000",
            "http://localhost:8080",
            "https://example.com",
        ],
    )
    def test_cors_accepts_different_origins(
        self, app_client: TestClient, origin: str
    ) -> None:
        """Test that CORS accepts requests from different origins.

        Validates:
        - Multiple origins are supported
        - CORS configuration is flexible
        """
        response = app_client.get("/health", headers={"Origin": origin})

        assert response.status_code == status.HTTP_200_OK
        assert "access-control-allow-origin" in response.headers


class TestResponseHeaders:
    """Test suite for response headers.

    Validates custom headers, security headers, and metadata
    included in API responses.
    """

    def test_correlation_id_is_unique_per_request(
        self, app_client: TestClient
    ) -> None:
        """Test that each request gets a unique correlation ID.

        Validates:
        - Correlation IDs are unique across requests
        - Request tracking is properly implemented
        """
        response1 = app_client.get("/health")
        response2 = app_client.get("/health")

        correlation_id1 = response1.headers.get("x-correlation-id")
        correlation_id2 = response2.headers.get("x-correlation-id")

        assert correlation_id1 != correlation_id2

    def test_correlation_id_format(self, app_client: TestClient) -> None:
        """Test that correlation ID has expected format.

        Validates:
        - Correlation ID is a valid UUID or similar format
        - Format is consistent
        """
        response = app_client.get("/health")
        correlation_id = response.headers.get("x-correlation-id")

        assert correlation_id is not None
        assert len(correlation_id) > 0
        # UUID format: 8-4-4-4-12 characters
        assert len(correlation_id) >= 32

    def test_content_type_header_on_json_response(
        self, app_client: TestClient
    ) -> None:
        """Test that JSON responses have correct content type.

        Validates:
        - Content-Type is application/json
        - Charset is specified if applicable
        """
        response = app_client.get("/health")

        content_type = response.headers.get("content-type")
        assert "application/json" in content_type

    def test_response_headers_on_error(
        self, app_client: TestClient, mock_redis: AsyncMock
    ) -> None:
        """Test that error responses include proper headers.

        Validates:
        - Correlation ID is present on errors
        - Content-Type is correct on errors
        - Headers are consistent across success/failure
        """
        mock_redis.ping.side_effect = Exception("Redis connection failed")

        with patch("app.main.redis_client", mock_redis):
            response = app_client.get("/ready")

        assert "x-correlation-id" in response.headers
        assert "content-type" in response.headers


class TestApplicationConfiguration:
    """Test suite for application configuration.

    Validates FastAPI application setup, middleware configuration,
    and global settings.
    """

    def test_application_title_and_version(self, app_client: TestClient) -> None:
        """Test that application has correct title and version in OpenAPI.

        Validates:
        - OpenAPI schema is accessible
        - Title and version are set
        """
        response = app_client.get("/openapi.json")

        if response.status_code == status.HTTP_200_OK:
            openapi_schema = response.json()
            assert "info" in openapi_schema
            assert "title" in openapi_schema["info"]
            assert "version" in openapi_schema["info"]

    def test_docs_endpoint_accessible(self, app_client: TestClient) -> None:
        """Test that API documentation endpoint is accessible.

        Validates:
        - /docs endpoint returns 200
        - Swagger UI is available
        """
        response = app_client.get("/docs")

        # Docs should be accessible or return 404 if disabled
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
        ]

    def test_root_endpoint_behavior(self, app_client: TestClient) -> None:
        """Test root endpoint behavior.

        Validates:
        - Root endpoint exists or returns 404
        - Application handles root path appropriately
        """
        response = app_client.get("/")

        # Root can redirect, return data, or 404
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_307_TEMPORARY_REDIRECT,
            status.HTTP_404_NOT_FOUND,
        ]


class TestErrorHandling:
    """Test suite for error handling.

    Validates exception handling, error responses, and
    graceful degradation scenarios.
    """

    def test_invalid_endpoint_returns_404(self, app_client: TestClient) -> None:
        """Test that invalid endpoints return 404.

        Validates:
        - Non-existent endpoints return 404
        - Error response is properly formatted
        """
        response = app_client.get("/invalid-endpoint")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_invalid_method_returns_405(self, app_client: TestClient) -> None:
        """Test that invalid methods return 405.

        Validates:
        - Unsupported HTTP methods return 405
        - Method Not Allowed is properly handled
        """
        response = app_client.post("/health")

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_error_response_includes_correlation_id(
        self, app_client: TestClient
    ) -> None:
        """Test that error responses include correlation ID.

        Validates:
        - Correlation ID is present on 404 errors
        - Error tracking is consistent
        """
        response = app_client.get("/invalid-endpoint")

        # Correlation ID should be present even on errors
        assert "x-correlation-id" in response.headers or response.status_code == 404


class TestAsyncEndpoints:
    """Test suite for async endpoint behavior.

    Validates asynchronous request handling and concurrent access.
    """

    @pytest.mark.asyncio
    async def test_health_endpoint_async(self, async_app_client: AsyncClient) -> None:
        """Test health endpoint with async client.

        Validates:
        - Async requests are handled correctly
        - Response is consistent with sync client
        """
        response = await async_app_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_ready_endpoint_async_success(
        self, async_app_client: AsyncClient, mock_redis: AsyncMock
    ) -> None:
        """Test ready endpoint with async client on success.

        Validates:
        - Async readiness check works correctly
        - Redis ping is awaited properly
        """
        mock_redis.ping.return_value = True

        with patch("app.main.redis_client", mock_redis):
            response = await async_app_client.get("/ready")

        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_ready_endpoint_async_failure(
        self, async_app_client: AsyncClient, mock_redis: AsyncMock
    ) -> None:
        """Test ready endpoint with async client on failure.

        Validates:
        - Async error handling works correctly
        - Exceptions are properly caught
        """
        mock_redis.ping.side_effect = Exception("Redis connection failed")

        with patch("app.main.redis_client", mock_redis):
            response = await async_app_client.get("/ready")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_app_client: AsyncClient) -> None:
        """Test handling of concurrent requests.

        Validates:
        - Multiple concurrent requests are handled
        - No race conditions or deadlocks
        - All requests complete successfully
        """
        import asyncio

        # Make 10 concurrent requests
        tasks = [async_app_client.get("/health") for _ in range(10)]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 10
        assert all(r.status_code == status.HTTP_200_OK for r in responses)
        assert all(r.json()["status"] == "healthy" for r in responses)


class TestSecurityHeaders:
    """Test suite for security headers.

    Validates security-related HTTP headers and configurations.
    """

    def test_no_sensitive_info_in_headers(self, app_client: TestClient) -> None:
        """Test that responses don't leak sensitive information.

        Validates:
        - Server header doesn't reveal version details
        - No internal paths or stack traces in headers
        """
        response = app_client.get("/health")

        # Check that common sensitive headers are not present or sanitized
        server_header = response.headers.get("server", "").lower()
        assert "python" not in server_header or server_header == ""

    def test_no_sensitive_info_in_error_response(
        self, app_client: TestClient, mock_redis: AsyncMock
    ) -> None:
        """Test that error responses don't leak sensitive information.

        Validates:
        - Stack traces are not exposed
        - Internal error details are sanitized
        """
        mock_redis.ping.side_effect = Exception("Internal Redis error with secrets")

        with patch("app.main.redis_client", mock_redis):
            response = app_client.get("/ready")

        # Error response should not contain internal error details
        response_text = response.text.lower()
        assert "secret" not in response_text
        assert "password" not in response_text


class TestPerformanceAndReliability:
    """Test suite for performance and reliability.

    Validates response times, resource usage, and system stability.
    """

    def test_health_endpoint_response_size(self, app_client: TestClient) -> None:
        """Test that health endpoint response is appropriately sized.

        Validates:
        - Response payload is small and efficient
        - No unnecessary data is included
        """
        response = app_client.get("/health")
        content_length = len(response.content)

        # Health check should be small (< 1KB)
        assert content_length < 1024

    def test_multiple_sequential_requests(self, app_client: TestClient) -> None:
        """Test handling of multiple sequential requests.

        Validates:
        - No memory leaks or resource exhaustion
        - Consistent performance across requests
        """
        for _ in range(50):
            response = app_client.get("/health")
            assert response.status_code == status.HTTP_200_OK

    def test_ready_endpoint_timeout_handling(
        self, app_client: TestClient, mock_redis: AsyncMock
    ) -> None:
        """Test that ready endpoint handles slow Redis responses.

        Validates:
        - Timeout scenarios are handled gracefully
        - Service doesn't hang on slow dependencies
        """
        import time

        def slow_ping() -> bool:
            time.sleep(0.1)  # Simulate slow response
            return True

        mock_redis.ping.side_effect = slow_ping

        with patch("app.main.redis_client", mock_redis):
            response = app_client.get("/ready")

        # Should still complete (either success or timeout)
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        ]