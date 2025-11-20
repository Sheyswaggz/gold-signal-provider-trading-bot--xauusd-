"""Pytest configuration and shared fixtures for the XAUUSD signal bot test suite.

This module provides test fixtures and configuration for pytest, including:
- FastAPI test client setup
- Mock Redis client for testing
- Test settings override
- Asyncio event loop configuration
- Test data cleanup utilities
"""

import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session.

    Yields:
        asyncio.AbstractEventLoop: Event loop for async tests
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings for testing.

    Returns:
        MagicMock: Mock settings object with test configuration
    """
    settings = MagicMock()
    settings.REDIS_HOST = "localhost"
    settings.REDIS_PORT = 6379
    settings.REDIS_DB = 0
    settings.REDIS_PASSWORD = None
    settings.REDIS_DECODE_RESPONSES = True
    settings.LOG_LEVEL = "INFO"
    settings.ENVIRONMENT = "test"
    return settings


@pytest_asyncio.fixture
async def mock_redis() -> AsyncMock:
    """Create a mock Redis client for testing.

    Returns:
        AsyncMock: Mock Redis client with common methods
    """
    redis_mock = AsyncMock()
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=0)
    redis_mock.expire = AsyncMock(return_value=True)
    redis_mock.close = AsyncMock()
    return redis_mock


@pytest.fixture
def app_client(mock_settings: MagicMock, mock_redis: AsyncMock) -> Generator[TestClient, None, None]:
    """Create a FastAPI test client with mocked dependencies.

    Args:
        mock_settings: Mock settings fixture
        mock_redis: Mock Redis client fixture

    Yields:
        TestClient: FastAPI test client for making requests
    """
    from app.main import app

    with TestClient(app) as client:
        yield client


@pytest_asyncio.fixture
async def async_app_client(
    mock_settings: MagicMock, mock_redis: AsyncMock
) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing FastAPI endpoints.

    Args:
        mock_settings: Mock settings fixture
        mock_redis: Mock Redis client fixture

    Yields:
        AsyncClient: Async HTTP client for making requests
    """
    from app.main import app

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(autouse=True)
def cleanup_test_data() -> Generator[None, None, None]:
    """Automatically cleanup test data after each test.

    This fixture runs before and after each test to ensure clean state.
    """
    yield
    # Cleanup logic runs after test completes


@pytest.fixture
def sample_test_data() -> dict:
    """Provide sample test data for tests.

    Returns:
        dict: Sample data structure for testing
    """
    return {
        "symbol": "XAUUSD",
        "price": 2000.50,
        "timestamp": "2024-01-01T00:00:00Z",
        "signal": "BUY",
    }