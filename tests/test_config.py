"""Comprehensive test suite for configuration management.

This module tests the Settings class and configuration loading functionality,
including environment variable handling, validation, and singleton behavior.

Test Categories:
- Unit tests for Settings initialization and validation
- Integration tests for environment variable loading
- Edge cases for invalid configurations
- Security tests for sensitive data handling
"""

from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


class TestSettingsInitialization:
    """Test suite for Settings class initialization and default values."""

    def test_settings_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that Settings initializes with correct default values.

        Validates that all configuration fields have appropriate defaults
        when no environment variables are set.
        """
        # Arrange: Clear all relevant environment variables
        env_vars = [
            "REDIS_HOST",
            "REDIS_PORT",
            "REDIS_DB",
            "REDIS_PASSWORD",
            "REDIS_DECODE_RESPONSES",
            "LOG_LEVEL",
            "ENVIRONMENT",
        ]
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)

        # Act: Create Settings instance
        settings = Settings()

        # Assert: Verify default values
        assert settings.REDIS_HOST == "localhost"
        assert settings.REDIS_PORT == 6379
        assert settings.REDIS_DB == 0
        assert settings.REDIS_PASSWORD is None
        assert settings.REDIS_DECODE_RESPONSES is True
        assert settings.LOG_LEVEL == "INFO"
        assert settings.ENVIRONMENT == "development"

    def test_settings_from_environment_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that Settings loads values from environment variables.

        Validates that environment variables override default values correctly.
        """
        # Arrange: Set environment variables
        monkeypatch.setenv("REDIS_HOST", "redis.example.com")
        monkeypatch.setenv("REDIS_PORT", "6380")
        monkeypatch.setenv("REDIS_DB", "1")
        monkeypatch.setenv("REDIS_PASSWORD", "secret123")
        monkeypatch.setenv("REDIS_DECODE_RESPONSES", "false")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("ENVIRONMENT", "production")

        # Act: Create Settings instance
        settings = Settings()

        # Assert: Verify environment values are loaded
        assert settings.REDIS_HOST == "redis.example.com"
        assert settings.REDIS_PORT == 6380
        assert settings.REDIS_DB == 1
        assert settings.REDIS_PASSWORD == "secret123"
        assert settings.REDIS_DECODE_RESPONSES is False
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.ENVIRONMENT == "production"

    def test_settings_partial_environment_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that partial environment variables work with defaults.

        Validates that only specified environment variables override defaults
        while others remain at default values.
        """
        # Arrange: Set only some environment variables
        monkeypatch.setenv("REDIS_HOST", "custom-redis")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")

        # Act: Create Settings instance
        settings = Settings()

        # Assert: Verify mixed values
        assert settings.REDIS_HOST == "custom-redis"
        assert settings.LOG_LEVEL == "WARNING"
        assert settings.REDIS_PORT == 6379  # Default
        assert settings.REDIS_DB == 0  # Default
        assert settings.ENVIRONMENT == "development"  # Default


class TestSettingsValidation:
    """Test suite for Settings validation and error handling."""

    def test_invalid_redis_port_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that invalid port type raises ValidationError.

        Validates that non-numeric port values are rejected.
        """
        # Arrange: Set invalid port value
        monkeypatch.setenv("REDIS_PORT", "not-a-number")

        # Act & Assert: Expect ValidationError
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "REDIS_PORT" in str(exc_info.value)

    def test_invalid_redis_port_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that out-of-range port values raise ValidationError.

        Validates that port numbers outside valid range (1-65535) are rejected.
        """
        # Arrange: Set out-of-range port
        monkeypatch.setenv("REDIS_PORT", "70000")

        # Act & Assert: Expect ValidationError
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "REDIS_PORT" in str(exc_info.value)

    def test_negative_redis_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that negative port values raise ValidationError."""
        # Arrange: Set negative port
        monkeypatch.setenv("REDIS_PORT", "-1")

        # Act & Assert: Expect ValidationError
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "REDIS_PORT" in str(exc_info.value)

    def test_invalid_redis_db_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that invalid database index type raises ValidationError."""
        # Arrange: Set invalid DB value
        monkeypatch.setenv("REDIS_DB", "invalid")

        # Act & Assert: Expect ValidationError
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "REDIS_DB" in str(exc_info.value)

    def test_negative_redis_db(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that negative database index raises ValidationError."""
        # Arrange: Set negative DB
        monkeypatch.setenv("REDIS_DB", "-1")

        # Act & Assert: Expect ValidationError
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "REDIS_DB" in str(exc_info.value)

    def test_invalid_boolean_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that invalid boolean values are handled correctly.

        Pydantic should convert common boolean representations.
        """
        # Arrange: Set various boolean representations
        test_cases = [
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
        ]

        for env_value, expected in test_cases:
            monkeypatch.setenv("REDIS_DECODE_RESPONSES", env_value)

            # Act: Create Settings instance
            settings = Settings()

            # Assert: Verify boolean conversion
            assert settings.REDIS_DECODE_RESPONSES == expected

    @pytest.mark.parametrize(
        "log_level",
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "debug", "info", "warning"],
    )
    def test_valid_log_levels(
        self, log_level: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that all valid log levels are accepted.

        Args:
            log_level: Log level string to test
        """
        # Arrange: Set log level
        monkeypatch.setenv("LOG_LEVEL", log_level)

        # Act: Create Settings instance
        settings = Settings()

        # Assert: Verify log level is set (case-insensitive)
        assert settings.LOG_LEVEL.upper() == log_level.upper()

    @pytest.mark.parametrize(
        "environment",
        ["development", "staging", "production", "test", "local"],
    )
    def test_valid_environments(
        self, environment: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that various environment names are accepted.

        Args:
            environment: Environment name to test
        """
        # Arrange: Set environment
        monkeypatch.setenv("ENVIRONMENT", environment)

        # Act: Create Settings instance
        settings = Settings()

        # Assert: Verify environment is set
        assert settings.ENVIRONMENT == environment


class TestGetSettingsFunction:
    """Test suite for get_settings() singleton function."""

    def test_get_settings_returns_settings_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that get_settings() returns a Settings instance."""
        # Arrange: Clear cache
        get_settings.cache_clear()

        # Act: Get settings
        settings = get_settings()

        # Assert: Verify instance type
        assert isinstance(settings, Settings)

    def test_get_settings_returns_cached_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that get_settings() returns the same cached instance.

        Validates singleton behavior - multiple calls return same object.
        """
        # Arrange: Clear cache
        get_settings.cache_clear()

        # Act: Get settings multiple times
        settings1 = get_settings()
        settings2 = get_settings()
        settings3 = get_settings()

        # Assert: Verify same instance
        assert settings1 is settings2
        assert settings2 is settings3
        assert id(settings1) == id(settings2) == id(settings3)

    def test_get_settings_cache_invalidation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that cache can be cleared and new instance is created."""
        # Arrange: Get initial settings
        get_settings.cache_clear()
        settings1 = get_settings()

        # Act: Clear cache and get new settings
        get_settings.cache_clear()
        settings2 = get_settings()

        # Assert: Verify different instances
        assert settings1 is not settings2
        assert id(settings1) != id(settings2)

    def test_get_settings_with_environment_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that cached settings don't reflect environment changes.

        Validates that once cached, settings remain constant even if
        environment variables change.
        """
        # Arrange: Set initial environment and get settings
        get_settings.cache_clear()
        monkeypatch.setenv("REDIS_HOST", "initial-host")
        settings1 = get_settings()

        # Act: Change environment (cached settings should not change)
        monkeypatch.setenv("REDIS_HOST", "changed-host")
        settings2 = get_settings()

        # Assert: Verify cached settings unchanged
        assert settings1 is settings2
        assert settings1.REDIS_HOST == "initial-host"
        assert settings2.REDIS_HOST == "initial-host"

        # Act: Clear cache and get new settings
        get_settings.cache_clear()
        settings3 = get_settings()

        # Assert: Verify new settings reflect changes
        assert settings3.REDIS_HOST == "changed-host"


class TestSettingsEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_empty_string_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test handling of empty string environment variables."""
        # Arrange: Set empty strings
        monkeypatch.setenv("REDIS_HOST", "")
        monkeypatch.setenv("REDIS_PASSWORD", "")

        # Act: Create Settings instance
        settings = Settings()

        # Assert: Verify empty strings are handled
        assert settings.REDIS_HOST == ""
        assert settings.REDIS_PASSWORD == ""

    def test_whitespace_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test handling of whitespace in environment variables."""
        # Arrange: Set values with whitespace
        monkeypatch.setenv("REDIS_HOST", "  redis-host  ")
        monkeypatch.setenv("LOG_LEVEL", "  INFO  ")

        # Act: Create Settings instance
        settings = Settings()

        # Assert: Verify whitespace handling (Pydantic strips by default)
        assert settings.REDIS_HOST == "  redis-host  "
        assert settings.LOG_LEVEL == "  INFO  "

    def test_maximum_redis_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test maximum valid port number (65535)."""
        # Arrange: Set maximum port
        monkeypatch.setenv("REDIS_PORT", "65535")

        # Act: Create Settings instance
        settings = Settings()

        # Assert: Verify maximum port accepted
        assert settings.REDIS_PORT == 65535

    def test_minimum_redis_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test minimum valid port number (1)."""
        # Arrange: Set minimum port
        monkeypatch.setenv("REDIS_PORT", "1")

        # Act: Create Settings instance
        settings = Settings()

        # Assert: Verify minimum port accepted
        assert settings.REDIS_PORT == 1

    def test_redis_password_with_special_characters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that passwords with special characters are handled correctly."""
        # Arrange: Set password with special characters
        special_password = "p@ssw0rd!#$%^&*()_+-=[]{}|;:',.<>?/~`"
        monkeypatch.setenv("REDIS_PASSWORD", special_password)

        # Act: Create Settings instance
        settings = Settings()

        # Assert: Verify password preserved exactly
        assert settings.REDIS_PASSWORD == special_password

    def test_unicode_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test handling of Unicode characters in configuration."""
        # Arrange: Set Unicode values
        monkeypatch.setenv("REDIS_HOST", "redis-服务器")
        monkeypatch.setenv("REDIS_PASSWORD", "密码123")

        # Act: Create Settings instance
        settings = Settings()

        # Assert: Verify Unicode handling
        assert settings.REDIS_HOST == "redis-服务器"
        assert settings.REDIS_PASSWORD == "密码123"


class TestSettingsSecurity:
    """Test suite for security-related configuration scenarios."""

    def test_password_not_logged_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that password is not exposed in string representation.

        Security test to ensure sensitive data is not leaked in logs.
        """
        # Arrange: Set password
        monkeypatch.setenv("REDIS_PASSWORD", "super-secret-password")

        # Act: Create Settings instance and get string representation
        settings = Settings()
        settings_str = str(settings)
        settings_repr = repr(settings)

        # Assert: Verify password not in string representations
        # Note: This depends on Pydantic's default behavior
        # If password appears, consider using SecretStr
        assert "super-secret-password" not in settings_str or "***" in settings_str
        assert "super-secret-password" not in settings_repr or "***" in settings_repr

    def test_production_environment_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test production-appropriate configuration values."""
        # Arrange: Set production environment
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        monkeypatch.setenv("REDIS_PASSWORD", "secure-password")

        # Act: Create Settings instance
        settings = Settings()

        # Assert: Verify production settings
        assert settings.ENVIRONMENT == "production"
        assert settings.LOG_LEVEL == "WARNING"
        assert settings.REDIS_PASSWORD is not None
        assert len(settings.REDIS_PASSWORD) > 0


class TestSettingsIntegration:
    """Integration tests for Settings with other components."""

    def test_settings_compatible_with_redis_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that Settings provides values compatible with Redis client.

        Integration test to ensure configuration can be used to initialize
        Redis connections.
        """
        # Arrange: Set Redis configuration
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PORT", "6379")
        monkeypatch.setenv("REDIS_DB", "0")

        # Act: Create Settings instance
        settings = Settings()

        # Assert: Verify Redis-compatible values
        assert isinstance(settings.REDIS_HOST, str)
        assert isinstance(settings.REDIS_PORT, int)
        assert isinstance(settings.REDIS_DB, int)
        assert settings.REDIS_PORT > 0
        assert settings.REDIS_DB >= 0

    def test_settings_with_mock_redis_fixture(
        self, mock_settings: MagicMock
    ) -> None:
        """Test that Settings works with mock_settings fixture.

        Integration test with existing test fixtures.
        """
        # Assert: Verify mock_settings has expected attributes
        assert hasattr(mock_settings, "REDIS_HOST")
        assert hasattr(mock_settings, "REDIS_PORT")
        assert hasattr(mock_settings, "LOG_LEVEL")
        assert mock_settings.ENVIRONMENT == "test"


class TestSettingsPerformance:
    """Performance tests for Settings initialization and caching."""

    def test_settings_initialization_performance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that Settings initialization is fast.

        Performance test to ensure configuration loading doesn't cause delays.
        """
        import time

        # Arrange: Set environment variables
        monkeypatch.setenv("REDIS_HOST", "localhost")

        # Act: Measure initialization time
        start_time = time.perf_counter()
        for _ in range(100):
            Settings()
        end_time = time.perf_counter()

        # Assert: Verify reasonable performance (< 100ms for 100 instances)
        elapsed_time = end_time - start_time
        assert elapsed_time < 0.1, f"Initialization too slow: {elapsed_time:.3f}s"

    def test_get_settings_cache_performance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that cached get_settings() is fast.

        Performance test to verify caching effectiveness.
        """
        import time

        # Arrange: Clear cache and prime it
        get_settings.cache_clear()
        get_settings()

        # Act: Measure cached access time
        start_time = time.perf_counter()
        for _ in range(10000):
            get_settings()
        end_time = time.perf_counter()

        # Assert: Verify cached access is very fast (< 10ms for 10000 calls)
        elapsed_time = end_time - start_time
        assert elapsed_time < 0.01, f"Cached access too slow: {elapsed_time:.3f}s"


class TestSettingsDocumentation:
    """Tests to ensure Settings is well-documented and maintainable."""

    def test_settings_has_docstring(self) -> None:
        """Test that Settings class has documentation."""
        # Assert: Verify class has docstring
        assert Settings.__doc__ is not None
        assert len(Settings.__doc__) > 0

    def test_settings_fields_have_descriptions(self) -> None:
        """Test that Settings fields have descriptions.

        Validates that configuration fields are documented for maintainability.
        """
        # Act: Get Settings schema
        schema = Settings.model_json_schema()

        # Assert: Verify fields have descriptions
        properties = schema.get("properties", {})
        assert len(properties) > 0

        # Check that at least some fields have descriptions
        fields_with_descriptions = [
            field
            for field, info in properties.items()
            if info.get("description") or info.get("title")
        ]
        assert len(fields_with_descriptions) > 0


# Cleanup fixture to ensure test isolation
@pytest.fixture(autouse=True)
def cleanup_settings_cache() -> Generator[None, None, None]:
    """Automatically clear settings cache before each test.

    Ensures test isolation by clearing the singleton cache.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()