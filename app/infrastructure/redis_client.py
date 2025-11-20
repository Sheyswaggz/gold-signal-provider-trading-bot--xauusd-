"""Redis client wrapper for caching and pub/sub with connection pooling.

This module provides an async Redis client with connection pooling, retry logic,
and health checking capabilities for production use.
"""

import asyncio
import logging
from typing import Any, Optional

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis client with connection pooling and retry logic.

    Provides methods for caching operations (get, set, delete) and pub/sub
    functionality with automatic reconnection and health checking.
    """

    def __init__(self) -> None:
        """Initialize Redis client with connection pool configuration."""
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._max_retries: int = 3
        self._base_backoff: float = 0.5

    async def connect(self) -> None:
        """Establish connection to Redis with connection pooling.

        Creates a connection pool with max_connections=10 and initializes
        the Redis client. Implements exponential backoff retry logic for
        connection failures.

        Raises:
            ConnectionError: If unable to connect after max retries.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                self._pool = ConnectionPool.from_url(
                    settings.redis_url,
                    max_connections=10,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    health_check_interval=30,
                )
                self._client = redis.Redis(connection_pool=self._pool)

                # Verify connection
                await self._client.ping()

                logger.info(
                    "Redis connection established",
                    extra={
                        "redis_url": settings.redis_url,
                        "max_connections": 10,
                        "attempt": attempt + 1,
                    },
                )
                return

            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                backoff_time = self._base_backoff * (2**attempt)

                logger.warning(
                    "Redis connection attempt failed",
                    extra={
                        "attempt": attempt + 1,
                        "max_retries": self._max_retries,
                        "backoff_seconds": backoff_time,
                        "error": str(e),
                    },
                )

                if attempt < self._max_retries - 1:
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(
                        "Redis connection failed after all retries",
                        extra={
                            "max_retries": self._max_retries,
                            "error": str(e),
                        },
                    )

        raise ConnectionError(
            f"Failed to connect to Redis after {self._max_retries} attempts"
        ) from last_exception

    async def disconnect(self) -> None:
        """Close Redis connection and cleanup resources.

        Gracefully closes the Redis client connection and releases
        connection pool resources.
        """
        if self._client:
            try:
                await self._client.aclose()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(
                    "Error closing Redis connection",
                    extra={"error": str(e)},
                )
            finally:
                self._client = None

        if self._pool:
            try:
                await self._pool.aclose()
            except Exception as e:
                logger.error(
                    "Error closing Redis connection pool",
                    extra={"error": str(e)},
                )
            finally:
                self._pool = None

    async def ping(self) -> bool:
        """Check if Redis connection is alive.

        Returns:
            bool: True if connection is healthy, False otherwise.
        """
        if not self._client:
            return False

        try:
            await self._client.ping()
            return True
        except (ConnectionError, TimeoutError) as e:
            logger.warning(
                "Redis ping failed",
                extra={"error": str(e)},
            )
            return False

    async def get(self, key: str) -> Optional[str]:
        """Retrieve value from Redis by key.

        Args:
            key: Redis key to retrieve.

        Returns:
            Optional[str]: Value if key exists, None otherwise.

        Raises:
            ConnectionError: If Redis connection is not established.
        """
        if not self._client:
            raise ConnectionError("Redis client not connected")

        try:
            value = await self._client.get(key)
            logger.debug(
                "Redis GET operation",
                extra={"key": key, "found": value is not None},
            )
            return value
        except (ConnectionError, TimeoutError) as e:
            logger.error(
                "Redis GET operation failed",
                extra={"key": key, "error": str(e)},
            )
            raise

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set key-value pair in Redis with optional TTL.

        Args:
            key: Redis key to set.
            value: Value to store.
            ttl: Time-to-live in seconds. None for no expiration.

        Returns:
            bool: True if operation succeeded, False otherwise.

        Raises:
            ConnectionError: If Redis connection is not established.
        """
        if not self._client:
            raise ConnectionError("Redis client not connected")

        try:
            if ttl is not None:
                result = await self._client.setex(key, ttl, value)
            else:
                result = await self._client.set(key, value)

            logger.debug(
                "Redis SET operation",
                extra={"key": key, "ttl": ttl, "success": bool(result)},
            )
            return bool(result)
        except (ConnectionError, TimeoutError) as e:
            logger.error(
                "Redis SET operation failed",
                extra={"key": key, "ttl": ttl, "error": str(e)},
            )
            raise

    async def delete(self, key: str) -> int:
        """Delete key from Redis.

        Args:
            key: Redis key to delete.

        Returns:
            int: Number of keys deleted (0 or 1).

        Raises:
            ConnectionError: If Redis connection is not established.
        """
        if not self._client:
            raise ConnectionError("Redis client not connected")

        try:
            result = await self._client.delete(key)
            logger.debug(
                "Redis DELETE operation",
                extra={"key": key, "deleted": result},
            )
            return result
        except (ConnectionError, TimeoutError) as e:
            logger.error(
                "Redis DELETE operation failed",
                extra={"key": key, "error": str(e)},
            )
            raise

    async def publish(self, channel: str, message: str) -> int:
        """Publish message to Redis pub/sub channel.

        Args:
            channel: Channel name to publish to.
            message: Message to publish.

        Returns:
            int: Number of subscribers that received the message.

        Raises:
            ConnectionError: If Redis connection is not established.
        """
        if not self._client:
            raise ConnectionError("Redis client not connected")

        try:
            result = await self._client.publish(channel, message)
            logger.debug(
                "Redis PUBLISH operation",
                extra={"channel": channel, "subscribers": result},
            )
            return result
        except (ConnectionError, TimeoutError) as e:
            logger.error(
                "Redis PUBLISH operation failed",
                extra={"channel": channel, "error": str(e)},
            )
            raise

    async def is_healthy(self) -> bool:
        """Check if Redis client is healthy and connected.

        Performs a ping operation to verify connection health.

        Returns:
            bool: True if client is healthy and connected, False otherwise.
        """
        if not self._client or not self._pool:
            logger.warning("Redis health check failed: client not initialized")
            return False

        try:
            await self._client.ping()
            logger.debug("Redis health check passed")
            return True
        except (ConnectionError, TimeoutError) as e:
            logger.warning(
                "Redis health check failed",
                extra={"error": str(e)},
            )
            return False
        except Exception as e:
            logger.error(
                "Unexpected error during Redis health check",
                extra={"error": str(e)},
            )
            return False